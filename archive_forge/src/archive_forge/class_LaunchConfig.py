import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch.distributed.elastic import events, metrics
from torch.distributed.elastic.agent.server.api import WorkerSpec
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.multiprocessing import SignalException, Std
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint
from torch.distributed.elastic.utils.logging import get_logger
@dataclass
class LaunchConfig:
    """
    Creates a rendezvous config.

    Args:
        min_nodes: Minimum amount of nodes that the user function will
                        be launched on. Elastic agent ensures that the user
                        function start only when the min_nodes amount enters
                        the rendezvous.
        max_nodes: Maximum amount of nodes that the user function
                        will be launched on.
        nproc_per_node: On each node the elastic agent will launch
                            this amount of workers that will execute user
                            defined function.
        rdzv_backend: rdzv_backend to use in the rendezvous (zeus-adapter, etcd).
        rdzv_endpoint: The endpoint of the rdzv sync. storage.
        rdzv_configs: Key, value pair that specifies rendezvous specific configuration.
        rdzv_timeout: Legacy argument that specifies timeout for the rendezvous. It is going
            to be removed in future versions, see the note below. The default timeout is 900 seconds.
        run_id: The unique run id of the job (if not passed a unique one will be
                deduced from run environment - flow workflow id in flow - or auto generated).
        role: User defined role of the worker (defaults to "trainer").
        max_restarts: The maximum amount of restarts that elastic agent will conduct
                    on workers before failure.
        monitor_interval: The interval in seconds that is used by the elastic_agent
                        as a period of monitoring workers.
        start_method: The method is used by the elastic agent to start the
                    workers (spawn, fork, forkserver).
        log_dir: base log directory where log files are written. If not set,
                one is created in a tmp dir but NOT removed on exit.
        redirects: configuration to redirect stdout/stderr to log files.
                Pass a single ``Std`` enum to redirect all workers,
                or a mapping keyed by local_rank to selectively redirect.
        tee: configuration to "tee" stdout/stderr to console + log file.
        metrics_cfg: configuration to initialize metrics.
        local_addr: address of the local node if any. If not set, a lookup on the local
                machine's FQDN will be performed.
    ..note:
        `rdzv_timeout` is a legacy argument that will be removed in future.
        Set the timeout via `rdzv_configs['timeout']`

    """
    min_nodes: int
    max_nodes: int
    nproc_per_node: int
    run_id: str = ''
    role: str = 'default_role'
    rdzv_endpoint: str = ''
    rdzv_backend: str = 'etcd'
    rdzv_configs: Dict[str, Any] = field(default_factory=dict)
    rdzv_timeout: int = -1
    max_restarts: int = 3
    monitor_interval: float = 30
    start_method: str = 'spawn'
    log_dir: Optional[str] = None
    log_line_prefix_template: Optional[str] = None
    redirects: Union[Std, Dict[int, Std]] = Std.NONE
    tee: Union[Std, Dict[int, Std]] = Std.NONE
    metrics_cfg: Dict[str, str] = field(default_factory=dict)
    local_addr: Optional[str] = None

    def __post_init__(self):
        default_timeout = 900
        if self.rdzv_timeout != -1:
            self.rdzv_configs['timeout'] = self.rdzv_timeout
        elif 'timeout' not in self.rdzv_configs:
            self.rdzv_configs['timeout'] = default_timeout