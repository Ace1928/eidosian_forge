import copy
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, Optional
from ray._private import ray_constants
from ray.autoscaler._private.command_runner import DockerCommandRunner, SSHCommandRunner
from ray.autoscaler._private.gcp.node import GCPTPUNode
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
class TPUVMSSHCommandRunner(SSHCommandRunner):
    """An SSH command runner with overwritten IP address calls."""

    def __init__(self, internal_ip: str, external_ip: str, worker_id: int, accelerator_type: str, *args, **kwargs):
        self._internal_ip = internal_ip
        self._external_ip = external_ip
        self._worker_id = worker_id
        self._accelerator_type = accelerator_type
        super().__init__(*args, **kwargs)

    def _get_node_ip(self) -> str:
        if self.use_internal_ip:
            return self._internal_ip
        else:
            return self._external_ip

    def run(self, cmd, timeout=120, exit_on_fail=False, port_forward=None, with_output=False, environment_variables: Dict[str, object]=None, run_env='auto', ssh_options_override_ssh_key='', shutdown_after_run=False) -> str:
        """Override the SSH run for TPU VM pods.

        Main functionality here we need to inject is to intercept the resources
        provided by the node_provider TPU node type fillout.

        node_provider will provide a resource "TPU-{TPU_POD_TYPE}-head" which:
        1) allows application developers to target worker 0 of an arbitary TPU pod, and
        2) signals to the autoscaler how to address the demand for more TPU pods.

        Without this intercept, then all workers of a TPU pod will have the
        "TPU-{TPU_POD_TYPE}-head" resource which will violate functionality (1)
        above.

        """
        if environment_variables:
            resources = environment_variables.get(ray_constants.RESOURCES_ENVIRONMENT_VARIABLE, None)
            if resources:
                if self._worker_id != 0:
                    tpu_pod_resource_type = f'TPU-{self._accelerator_type}-head'
                    if tpu_pod_resource_type in resources:
                        resources.pop(tpu_pod_resource_type, None)
                environment_variables[ray_constants.RESOURCES_ENVIRONMENT_VARIABLE] = resources
        return super().run(cmd=cmd, timeout=timeout, exit_on_fail=exit_on_fail, port_forward=port_forward, with_output=with_output, environment_variables=environment_variables, run_env=run_env, ssh_options_override_ssh_key=ssh_options_override_ssh_key, shutdown_after_run=shutdown_after_run)