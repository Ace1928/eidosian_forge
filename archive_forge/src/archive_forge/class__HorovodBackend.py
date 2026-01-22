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
class _HorovodBackend(Backend):
    share_cuda_visible_devices: bool = True

    def on_start(self, worker_group: WorkerGroup, backend_config: HorovodConfig):
        setup_futures = []
        for rank in range(len(worker_group)):
            worker_node_id = worker_group.workers[rank].metadata.node_id
            setup_futures.append(worker_group.execute_single_async(rank, _init_env_vars, rank, len(worker_group), worker_node_id))
        ray.get(setup_futures)
        self.coordinator = Coordinator(backend_config)
        node_ids = [w.metadata.node_id for w in worker_group.workers]
        hostnames = [w.metadata.hostname for w in worker_group.workers]
        for rank, (hostname, node_id) in enumerate(zip(hostnames, node_ids)):
            self.coordinator.register(hostname, node_id, rank)
        all_info = self.coordinator.finalize_registration()
        setup_futures = []
        for rank, local_cross_env_var in all_info.items():
            setup_futures.append(worker_group.execute_single_async(rank, update_env_vars, local_cross_env_var))
        ray.get(setup_futures)
        coordinator_envs = self.coordinator.establish_rendezvous()
        node_worker_indexes = [node_ids.index(node_id) for node_id in set(node_ids)]
        node_workers = [_HorovodWorkerWrapper(worker_group.workers[worker_index]) for worker_index in node_worker_indexes]
        assert len(node_workers) == len(self.coordinator.hostnames)
        nics = detect_nics(backend_config, all_host_names=list(self.coordinator.hostnames), node_workers=node_workers)
        coordinator_envs.update(nics_to_env_var(nics))
        worker_group.execute(update_env_vars, coordinator_envs)