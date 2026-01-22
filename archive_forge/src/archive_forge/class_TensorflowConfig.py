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
@PublicAPI(stability='beta')
@dataclass
class TensorflowConfig(BackendConfig):

    @property
    def backend_cls(self):
        return _TensorflowBackend