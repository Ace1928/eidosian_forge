import json
import logging
import os
from dataclasses import dataclass
from typing import List
import ray
from ray.train._internal.utils import get_address_and_port
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import Backend, BackendConfig
from ray.util import PublicAPI
class _TensorflowBackend(Backend):

    def on_start(self, worker_group: WorkerGroup, backend_config: TensorflowConfig):

        def get_url():
            address, port = get_address_and_port()
            return f'{address}:{port}'
        urls = worker_group.execute(get_url)
        setup_futures = []
        for i in range(len(worker_group)):
            setup_futures.append(worker_group.execute_single_async(i, _setup_tensorflow_environment, worker_addresses=urls, index=i))
        ray.get(setup_futures)