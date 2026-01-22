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
class RayFSDPStrategy(FSDPStrategy):
    """Subclass of FSDPStrategy to ensure compatibility with Ray orchestration.

    For a full list of initialization arguments, please refer to:
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        record_extra_usage_tag(TagKey.TRAIN_LIGHTNING_RAYFSDPSTRATEGY, '1')

    @property
    def root_device(self) -> torch.device:
        return get_worker_root_device()

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        return dict(num_replicas=self.world_size, rank=self.global_rank)

    def lightning_module_state_dict(self) -> Dict[str, Any]:
        """Gathers the full state dict to rank 0 on CPU."""
        assert self.model is not None, 'Failed to get the state dict for a None model!'
        if _LIGHTNING_GREATER_EQUAL_2_0 and _TORCH_FSDP_AVAILABLE:
            with FullyShardedDataParallel.state_dict_type(module=self.model, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                state_dict = self.model.state_dict()
                prefix_len = len('_forward_module.')
                return {k[prefix_len:]: v for k, v in state_dict.items()}
        else:
            return super().lightning_module_state_dict()