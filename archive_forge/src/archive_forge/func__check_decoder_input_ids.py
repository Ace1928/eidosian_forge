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
def _check_decoder_input_ids(prompt_ids, init_tokens, is_shortform, kwargs):
    decoder_input_ids = kwargs.get('decoder_input_ids', None)
    if prompt_ids is not None and decoder_input_ids is not None:
        raise ValueError(f'Cannot pass both `prompt_ids`: {prompt_ids} and `decoder_input_ids`: {decoder_input_ids}. Passing `decoder_input_ids` is deprecated, consider not passing it.')
    elif decoder_input_ids is not None and (not is_shortform):
        raise ValueError(f'Cannot pass both `decoder_input_ids`: {decoder_input_ids} for long-form generation. Consider passing `prompt_ids` instead.')
    elif decoder_input_ids is not None and is_shortform:
        warnings.warn(f'You have provided `decoder_input_ids` which will overwrite the `init_tokens` {init_tokens}. This might lead to unexpected behavior. Passing `decoder_input_ids` is deprecated and will be removed in v4.39. Consider passing `prompt_ids` instead.', FutureWarning)