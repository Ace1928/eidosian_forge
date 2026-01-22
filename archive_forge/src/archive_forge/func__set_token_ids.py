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
def _set_token_ids(generation_config, config, kwargs):
    eos_token_id = kwargs.pop('eos_token_id', None)
    decoder_start_token_id = kwargs.pop('decoder_start_token_id', None)
    eos_token_id = eos_token_id if eos_token_id is not None else generation_config.eos_token_id
    decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else generation_config.decoder_start_token_id
    generation_config.eos_token_id = eos_token_id if eos_token_id is not None else config.eos_token_id
    generation_config.decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else config.decoder_start_token_id