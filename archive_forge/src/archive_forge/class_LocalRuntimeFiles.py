from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import itertools
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import local
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.core import yaml
import six
class LocalRuntimeFiles(object):
    """Generates the developement environment files for a project."""

    def __init__(self, settings):
        """Initialize LocalRuntimeFiles.

    Args:
      settings: Local development settings.
    """
        self._settings = settings

    def KubernetesConfig(self):
        """Create a kubernetes config file.

    Returns:
      Text of a kubernetes config file.
    """
        if self._settings.cpu:
            if isinstance(self._settings.cpu, six.text_type):
                if not self._settings.cpu.endswith('m'):
                    raise ValueError('cpu limit must be defined as an integer or as millicpus')
                user_cpu = int(self._settings.cpu[:-1]) / 1000.0
            else:
                user_cpu = self._settings.cpu
            cpu_request = min(0.1, user_cpu)
        else:
            cpu_request = None
        code_generators = [local.AppContainerGenerator(self._settings.service_name, self._settings.image, self._settings.env_vars, self._settings.env_vars_secrets, self._settings.memory, self._settings.cpu, cpu_request, self._settings.readiness_probe), local.SecretsGenerator(self._settings.service_name, self._settings.env_vars_secrets, self._settings.volumes_secrets, self._settings.namespace, self._settings.allow_secret_manager)]
        credential_generator = None
        if isinstance(self._settings.credential, local.ServiceAccountSetting):
            credential_generator = local.CredentialGenerator(functools.partial(local.GetServiceAccountSecret, self._settings.credential.name))
            code_generators.append(credential_generator)
        elif isinstance(self._settings.credential, local.ApplicationDefaultCredentialSetting):
            credential_generator = local.CredentialGenerator(local.GetUserCredential)
            code_generators.append(credential_generator)
        if self._settings.cloudsql_instances:
            if not credential_generator:
                raise ValueError('A credential generator must be defined when cloudsql instances are defined.')
            cloudsql_proxy = local.CloudSqlProxyGenerator(self._settings.cloudsql_instances, credential_generator.GetInfo())
            code_generators.append(cloudsql_proxy)
        return _GenerateKubeConfigs(code_generators)

    def SkaffoldConfig(self, kubernetes_file_path):
        """Create a skaffold yaml file.

    Args:
      kubernetes_file_path: Path to the kubernetes config file.

    Returns:
      Text of the skaffold yaml file.
    """
        skaffold_yaml = yaml.load(_SKAFFOLD_TEMPLATE)
        manifests = yaml_helper.GetOrCreate(skaffold_yaml, ('deploy', 'kubectl', 'manifests'), constructor=list)
        manifests.append(kubernetes_file_path)
        artifact = {'image': self._settings.image}
        artifact['context'] = six.ensure_text(self._settings.context.encode('unicode_escape'))
        if isinstance(self._settings.builder, builders.BuildpackBuilder):
            artifact['buildpacks'] = {'builder': self._settings.builder.builder}
            if self._settings.builder.devmode:
                artifact['buildpacks']['env'] = ['GOOGLE_DEVMODE=1']
                artifact['sync'] = {'auto': {}}
            if self._settings.builder.trust:
                artifact['buildpacks']['trustBuilder'] = True
        else:
            dockerfile_rel_path = self._settings.builder.DockerfileRelPath(self._settings.context)
            artifact['docker'] = {'dockerfile': six.ensure_text(dockerfile_rel_path.encode('unicode_escape'))}
        artifacts = yaml_helper.GetOrCreate(skaffold_yaml, ('build', 'artifacts'), constructor=list)
        artifacts.append(artifact)
        if self._settings.local_port:
            port_forward_config = {'resourceType': 'service', 'resourceName': self._settings.service_name, 'port': 8080, 'localPort': self._settings.local_port}
            if self._settings.namespace:
                port_forward_config['namespace'] = self._settings.namespace
            skaffold_yaml['portForward'] = [port_forward_config]
        return yaml.dump(skaffold_yaml)