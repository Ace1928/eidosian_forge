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
@dataclass
class _HorovodWorkerWrapper:
    w: Worker

    @property
    def execute(self):
        w = self.w

        class ExecuteHandle:

            def remote(self, func, *args, **kwargs):
                _ = None
                return w.actor._RayTrainWorker__execute.remote(func, _, *args, **kwargs)
        return ExecuteHandle()