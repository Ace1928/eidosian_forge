import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import PIL
import torch
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from .pipeline_utils import DiffusionPipelineMixin, rescale_noise_cfg
def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, aesthetic_score, negative_aesthetic_score, dtype):
    if self.config.get('requires_aesthetics_score'):
        add_time_ids = (original_size + crops_coords_top_left + (aesthetic_score,),)
        add_neg_time_ids = (original_size + crops_coords_top_left + (negative_aesthetic_score,),)
    else:
        add_time_ids = (original_size + crops_coords_top_left + target_size,)
        add_neg_time_ids = (original_size + crops_coords_top_left + target_size,)
    add_time_ids = np.array(add_time_ids, dtype=dtype)
    add_neg_time_ids = np.array(add_neg_time_ids, dtype=dtype)
    return (add_time_ids, add_neg_time_ids)