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
def _set_max_new_tokens_and_length(config, decoder_input_ids, generation_config, kwargs):
    num_initial_tokens = min(config.max_target_positions // 2 - 1, decoder_input_ids.shape[-1] - 1)
    passed_max_length = kwargs.pop('max_length', None)
    passed_max_new_tokens = kwargs.pop('max_new_tokens', None)
    max_length_config = getattr(generation_config, 'max_length', None)
    max_new_tokens_config = getattr(generation_config, 'max_new_tokens', None)
    max_new_tokens = None
    max_length = None
    if passed_max_length is not None and passed_max_new_tokens is None:
        max_length = min(passed_max_length + num_initial_tokens, config.max_target_positions)
        logger.info(f'Increase max_length from {passed_max_length} to {max_length} since input is conditioned on previous segment.')
    elif max_length_config is not None and passed_max_new_tokens is None and (max_new_tokens_config is None):
        max_length = min(generation_config.max_length + num_initial_tokens, config.max_target_positions)
        logger.info(f'Increase max_length from {max_length_config} to {max_length} since input is conditioned on previous segment.')
    elif passed_max_new_tokens is not None and passed_max_new_tokens + decoder_input_ids.shape[-1] > config.max_target_positions:
        max_new_tokens = config.max_target_positions - decoder_input_ids.shape[-1]
    elif passed_max_new_tokens is None and max_new_tokens_config is not None and (max_new_tokens_config + decoder_input_ids.shape[-1] > config.max_target_positions):
        max_new_tokens = config.max_target_positions - decoder_input_ids.shape[-1]
    if max_new_tokens is not None:
        kwargs['max_new_tokens'] = max_new_tokens
    if max_length is not None:
        kwargs['max_length'] = max_length
    return kwargs