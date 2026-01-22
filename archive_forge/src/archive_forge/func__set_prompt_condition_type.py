import copy
import math
import warnings
import zlib
from typing import Callable, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import (
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
@staticmethod
def _set_prompt_condition_type(generation_config, prompt_condition_type):
    allowed_cond_types = ['first-segment', 'all-segments']
    prompt_condition_type = prompt_condition_type or allowed_cond_types[0]
    if prompt_condition_type not in allowed_cond_types:
        raise ValueError(f'`prompt_condition_type={prompt_condition_type} does not exist. Make sure to set `prompt_condition_type` to one of {', '.join(allowed_cond_types)}')
    if generation_config.condition_on_prev_tokens is not True and prompt_condition_type == 'all-segments':
        raise ValueError("Make sure to set `condition_on_prev_tokens=True` when setting `prompt_condition_type='all-segments'`.")
    generation_config.prompt_condition_type = prompt_condition_type