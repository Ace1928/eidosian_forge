import contextlib
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from ..core import randn_tensor
from ..import_utils import is_peft_available
from .sd_utils import convert_state_dict_to_diffusers
@dataclass
class DDPOSchedulerOutput:
    """
    Output class for the diffusers scheduler to be finetuned with the DDPO trainer

    Args:
        latents (`torch.Tensor`):
            Predicted sample at the previous timestep. Shape: `(batch_size, num_channels, height, width)`
        log_probs (`torch.Tensor`):
            Log probability of the above mentioned sample. Shape: `(batch_size)`
    """
    latents: torch.Tensor
    log_probs: torch.Tensor