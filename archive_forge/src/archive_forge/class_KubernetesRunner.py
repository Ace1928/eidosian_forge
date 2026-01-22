import asyncio
import base64
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import yaml
import wandb
from wandb.apis.internal import Api
from wandb.sdk.launch.agent.agent import LaunchAgent
from wandb.sdk.launch.environment.abstract import AbstractEnvironment
from wandb.sdk.launch.registry.abstract import AbstractRegistry
from wandb.sdk.launch.registry.azure_container_registry import AzureContainerRegistry
from wandb.sdk.launch.registry.local_registry import LocalRegistry
from wandb.sdk.launch.runner.abstract import Status
from wandb.sdk.launch.runner.kubernetes_monitor import (
from wandb.util import get_module
from .._project_spec import EntryPoint, LaunchProject
from ..builder.build import get_env_vars_dict
from ..errors import LaunchError
from ..utils import (
from .abstract import AbstractRun, AbstractRunner
import kubernetes_asyncio  # type: ignore # noqa: E402
from kubernetes_asyncio import client  # noqa: E402
from kubernetes_asyncio.client.api.batch_v1_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.api.core_v1_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.api.custom_objects_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.models.v1_secret import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.rest import ApiException  # type: ignore # noqa: E402
class KubernetesRunner(AbstractRunner):
    """Launches runs onto kubernetes."""

    def __init__(self, api: Api, backend_config: Dict[str, Any], environment: AbstractEnvironment, registry: AbstractRegistry) -> None:
        """Create a Kubernetes runner.

        Arguments:
            api: The API client object.
            backend_config: The backend configuration.
            environment: The environment to launch runs into.

        Raises:
            LaunchError: If the Kubernetes configuration is invalid.
        """
        super().__init__(api, backend_config)
        self.environment = environment
        self.registry = registry

    def get_namespace(self, resource_args: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Get the namespace to launch into.

        Arguments:
            resource_args: The resource args to launch.
            context: The k8s config context.

        Returns:
            The namespace to launch into.
        """
        default_namespace = context['context'].get('namespace', 'default') if context else 'default'
        return resource_args.get('metadata', {}).get('namespace') or resource_args.get('namespace') or self.backend_config.get('runner', {}).get('namespace') or default_namespace

    async def _inject_defaults(self, resource_args: Dict[str, Any], launch_project: LaunchProject, image_uri: str, namespace: str, core_api: 'CoreV1Api') -> Tuple[Dict[str, Any], Optional['V1Secret']]:
        """Apply our default values, return job dict and api key secret.

        Arguments:
            resource_args (Dict[str, Any]): The resource args to launch.
            launch_project (LaunchProject): The launch project.
            builder (Optional[AbstractBuilder]): The builder.
            namespace (str): The namespace.
            core_api (CoreV1Api): The core api.

        Returns:
            Tuple[Dict[str, Any], Optional["V1Secret"]]: The resource args and api key secret.
        """
        job: Dict[str, Any] = {'apiVersion': 'batch/v1', 'kind': 'Job'}
        job.update(resource_args)
        job_metadata: Dict[str, Any] = job.get('metadata', {})
        job_spec: Dict[str, Any] = {'backoffLimit': 0, 'ttlSecondsAfterFinished': 60}
        job_spec.update(job.get('spec', {}))
        pod_template: Dict[str, Any] = job_spec.get('template', {})
        pod_spec: Dict[str, Any] = {'restartPolicy': 'Never'}
        pod_spec.update(pod_template.get('spec', {}))
        containers: List[Dict[str, Any]] = pod_spec.get('containers', [{}])
        job_metadata.setdefault('labels', {})
        job_metadata['labels'][WANDB_K8S_RUN_ID] = launch_project.run_id
        job_metadata['labels'][WANDB_K8S_LABEL_MONITOR] = 'true'
        if LaunchAgent.initialized():
            job_metadata['labels'][WANDB_K8S_LABEL_AGENT] = LaunchAgent.name()
        if not job_metadata.get('name'):
            job_metadata['generateName'] = make_name_dns_safe(f'launch-{launch_project.target_entity}-{launch_project.target_project}-')
        for i, cont in enumerate(containers):
            if 'name' not in cont:
                cont['name'] = cont.get('name', 'launch' + str(i))
            if 'securityContext' not in cont:
                cont['securityContext'] = {'allowPrivilegeEscalation': False, 'capabilities': {'drop': ['ALL']}, 'seccompProfile': {'type': 'RuntimeDefault'}}
        entry_point = launch_project.override_entrypoint or launch_project.get_single_entry_point()
        if launch_project.docker_image:
            containers[0]['image'] = image_uri
        elif not any(['image' in cont for cont in containers]):
            assert entry_point is not None
            containers[0]['image'] = image_uri
        secret = await maybe_create_imagepull_secret(core_api, self.registry, launch_project.run_id, namespace)
        if secret is not None:
            pod_spec['imagePullSecrets'] = [{'name': f'regcred-{launch_project.run_id}'}]
        inject_entrypoint_and_args(containers, entry_point, launch_project.override_args, launch_project.override_entrypoint is not None)
        env_vars = get_env_vars_dict(launch_project, self._api, MAX_ENV_LENGTHS[self.__class__.__name__])
        api_key_secret = None
        for cont in containers:
            env = cont.get('env') or []
            for key, value in env_vars.items():
                if key == 'WANDB_API_KEY' and value and (LaunchAgent.initialized() or self.backend_config[PROJECT_SYNCHRONOUS]):
                    release_name = os.environ.get('WANDB_RELEASE_NAME')
                    secret_name = 'wandb-api-key'
                    if release_name:
                        secret_name += f'-{release_name}'
                    else:
                        secret_name += f'-{launch_project.run_id}'
                    api_key_secret = await ensure_api_key_secret(core_api, secret_name, namespace, value)
                    env.append({'name': key, 'valueFrom': {'secretKeyRef': {'name': secret_name, 'key': 'password'}}})
                else:
                    env.append({'name': key, 'value': value})
            cont['env'] = env
        pod_spec['containers'] = containers
        pod_template['spec'] = pod_spec
        job_spec['template'] = pod_template
        job['spec'] = job_spec
        job['metadata'] = job_metadata
        add_label_to_pods(job, WANDB_K8S_LABEL_MONITOR, 'true')
        if LaunchAgent.initialized():
            add_label_to_pods(job, WANDB_K8S_LABEL_AGENT, LaunchAgent.name())
        return (job, api_key_secret)

    async def run(self, launch_project: LaunchProject, image_uri: str) -> Optional[AbstractRun]:
        """Execute a launch project on Kubernetes.

        Arguments:
            launch_project: The launch project to execute.
            builder: The builder to use to build the image.

        Returns:
            The run object if the run was successful, otherwise None.
        """
        await LaunchKubernetesMonitor.ensure_initialized()
        resource_args = launch_project.fill_macros(image_uri).get('kubernetes', {})
        if not resource_args:
            wandb.termlog(f'{LOG_PREFIX}Note: no resource args specified. Add a Kubernetes yaml spec or other options in a json file with --resource-args <json>.')
        _logger.info(f'Running Kubernetes job with resource args: {resource_args}')
        context, api_client = await get_kube_context_and_api_client(kubernetes_asyncio, resource_args)
        api_version = resource_args.get('apiVersion', 'batch/v1')
        if api_version not in ['batch/v1', 'batch/v1beta1']:
            env_vars = get_env_vars_dict(launch_project, self._api, MAX_ENV_LENGTHS[self.__class__.__name__])
            add_wandb_env(resource_args, env_vars)
            resource_args['metadata'] = resource_args.get('metadata', {})
            resource_args['metadata']['labels'] = resource_args['metadata'].get('labels', {})
            resource_args['metadata']['labels'][WANDB_K8S_LABEL_MONITOR] = 'true'
            add_label_to_pods(resource_args, WANDB_K8S_LABEL_MONITOR, 'true')
            if LaunchAgent.initialized():
                add_label_to_pods(resource_args, WANDB_K8S_LABEL_MONITOR, LaunchAgent.name())
                resource_args['metadata']['labels'][WANDB_K8S_LABEL_AGENT] = LaunchAgent.name()
            overrides = {}
            if launch_project.override_args:
                overrides['args'] = launch_project.override_args
            if launch_project.override_entrypoint:
                overrides['command'] = launch_project.override_entrypoint.command
            add_entrypoint_args_overrides(resource_args, overrides)
            api = client.CustomObjectsApi(api_client)
            namespace = self.get_namespace(resource_args, context)
            group, version, *_ = api_version.split('/')
            group = resource_args.get('group', group)
            version = resource_args.get('version', version)
            kind = resource_args.get('kind', version)
            plural = f'{kind.lower()}s'
            custom_resource = CustomResource(group=group, version=version, plural=plural)
            LaunchKubernetesMonitor.monitor_namespace(namespace, custom_resource=custom_resource)
            try:
                response = await api.create_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural, body=resource_args)
            except ApiException as e:
                body = json.loads(e.body)
                body_yaml = yaml.dump(body)
                raise LaunchError(f'Error creating CRD of kind {kind}: {e.status} {e.reason}\n{body_yaml}') from e
            name = response.get('metadata', {}).get('name')
            _logger.info(f'Created {kind} {response['metadata']['name']}')
            submitted_run = CrdSubmittedRun(name=name, group=group, version=version, namespace=namespace, plural=plural, core_api=client.CoreV1Api(api_client), custom_api=api)
            if self.backend_config[PROJECT_SYNCHRONOUS]:
                await submitted_run.wait()
            return submitted_run
        batch_api = kubernetes_asyncio.client.BatchV1Api(api_client)
        core_api = kubernetes_asyncio.client.CoreV1Api(api_client)
        namespace = self.get_namespace(resource_args, context)
        job, secret = await self._inject_defaults(resource_args, launch_project, image_uri, namespace, core_api)
        msg = 'Creating Kubernetes job'
        if 'name' in resource_args:
            msg += f': {resource_args['name']}'
        _logger.info(msg)
        try:
            response = await kubernetes_asyncio.utils.create_from_dict(api_client, job, namespace=namespace)
        except kubernetes_asyncio.utils.FailToCreateError as e:
            for exc in e.api_exceptions:
                resp = json.loads(exc.body)
                msg = resp.get('message')
                code = resp.get('code')
                raise LaunchError(f'Failed to create Kubernetes job for run {launch_project.run_id} ({code} {exc.reason}): {msg}')
        except Exception as e:
            raise LaunchError(f'Unexpected exception when creating Kubernetes job: {str(e)}\n')
        job_response = response[0]
        job_name = job_response.metadata.name
        LaunchKubernetesMonitor.monitor_namespace(namespace)
        submitted_job = KubernetesSubmittedRun(batch_api, core_api, job_name, namespace, secret)
        if self.backend_config[PROJECT_SYNCHRONOUS]:
            await submitted_job.wait()
        return submitted_job