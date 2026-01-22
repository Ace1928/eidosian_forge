import logging
import os
import shutil
import tempfile
from typing import Any, Dict
import torch
from packaging.version import Version
import ray
from ray import train
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train import Checkpoint
from ray.util import PublicAPI
@PublicAPI(stability='beta')
class RayLightningEnvironment(LightningEnvironment):
    """Setup Lightning DDP training environment for Ray cluster."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        record_extra_usage_tag(TagKey.TRAIN_LIGHTNING_RAYLIGHTNINGENVIRONMENT, '1')

    def world_size(self) -> int:
        return train.get_context().get_world_size()

    def global_rank(self) -> int:
        return train.get_context().get_world_rank()

    def local_rank(self) -> int:
        return train.get_context().get_local_rank()

    def node_rank(self) -> int:
        return train.get_context().get_node_rank()

    def set_world_size(self, size: int) -> None:
        pass

    def set_global_rank(self, rank: int) -> None:
        pass

    def teardown(self):
        pass