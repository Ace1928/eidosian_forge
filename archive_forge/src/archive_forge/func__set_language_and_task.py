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
def _set_language_and_task(language, task, is_multilingual, generation_config):
    if is_multilingual is not None:
        if not hasattr(generation_config, 'is_multilingual'):
            raise ValueError('The generation config is outdated and is thus not compatible with the `is_multilingual` argument to `generate`. Please update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224')
        generation_config.is_multilingual = is_multilingual
    if hasattr(generation_config, 'is_multilingual') and (not generation_config.is_multilingual):
        if task is not None or language is not None:
            raise ValueError('Cannot specify `task` or `language` for an English-only model. If the model is intended to be multilingual, pass `is_multilingual=True` to generate, or update the generation config.')
    if language is not None:
        if not hasattr(generation_config, 'lang_to_id'):
            raise ValueError('The generation config is outdated and is thus not compatible with the `language` argument to `generate`. Either set the language using the `forced_decoder_ids` in the model config, or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224')
        language = language.lower()
        generation_config.language = language
    if task is not None:
        if not hasattr(generation_config, 'task_to_id'):
            raise ValueError('The generation config is outdated and is thus not compatible with the `task` argument to `generate`. Either set the task using the `forced_decoder_ids` in the model config, or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224')
        generation_config.task = task