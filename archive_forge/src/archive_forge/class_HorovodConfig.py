import os
from dataclasses import dataclass
from typing import Optional, Set
from horovod.ray.runner import Coordinator
from horovod.ray.utils import detect_nics, nics_to_env_var
from horovod.runner.common.util import secret, timeout
import ray
from ray.train._internal.utils import update_env_vars
from ray.train._internal.worker_group import Worker, WorkerGroup
from ray.train.backend import Backend, BackendConfig
from ray.util import PublicAPI
@PublicAPI(stability='beta')
@dataclass
class HorovodConfig(BackendConfig):
    """Configurations for Horovod setup.

    See https://github.com/horovod/horovod/blob/master/horovod/runner/common/util/settings.py # noqa: E501

    Args:
        nics (Optional[Set[str]): Network interfaces that can be used for
            communication.
        verbose: Horovod logging verbosity.
        key (Optional[str]): Secret used for communication between workers.
        ssh_port (Optional[int]): Port for SSH server running on worker nodes.
        ssh_identity_file (Optional[str]): Path to the identity file to
            ssh into different hosts on the cluster.
        ssh_str (Optional[str]): CAUTION WHEN USING THIS. Private key
            file contents. Writes the private key to ssh_identity_file.
        timeout_s: Timeout parameter for Gloo rendezvous.
        placement_group_timeout_s: Timeout parameter for Ray
            Placement Group creation. Currently unused.
    """
    nics: Optional[Set[str]] = None
    verbose: int = 1
    key: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_identity_file: Optional[str] = None
    ssh_str: Optional[str] = None
    timeout_s: int = 300
    placement_group_timeout_s: int = 100

    @property
    def start_timeout(self):
        return timeout.Timeout(self.timeout_s, message='Timed out waiting for {activity}. Please check connectivity between servers. You may need to increase the --start-timeout parameter if you have too many servers.')

    def __post_init__(self):
        if self.ssh_str and (not os.path.exists(self.ssh_identity_file)):
            with open(self.ssh_identity_file, 'w') as f:
                os.chmod(self.ssh_identity_file, 384)
                f.write(self.ssh_str)
        if self.key is None:
            self.key = secret.make_secret_key()

    @property
    def backend_cls(self):
        return _HorovodBackend