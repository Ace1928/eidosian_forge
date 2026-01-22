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
def _set_thresholds_and_condition(generation_config, logprob_threshold, compression_ratio_threshold, no_speech_threshold, condition_on_prev_tokens):
    generation_config.logprob_threshold = logprob_threshold if logprob_threshold is not None else getattr(generation_config, 'logprob_threshold', None)
    generation_config.compression_ratio_threshold = compression_ratio_threshold if compression_ratio_threshold is not None else getattr(generation_config, 'compression_ratio_threshold', None)
    generation_config.no_speech_threshold = no_speech_threshold if no_speech_threshold is not None else getattr(generation_config, 'no_speech_threshold', None)
    generation_config.condition_on_prev_tokens = condition_on_prev_tokens if condition_on_prev_tokens is not None else getattr(generation_config, 'condition_on_prev_tokens', None)