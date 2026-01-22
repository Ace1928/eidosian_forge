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
def _get_attr_from_logit_processors(logits_processor, logit_processor_class, attribute_name):
    logit_processor = next((cls for cls in logits_processor if isinstance(cls, logit_processor_class)), None)
    if logit_processor:
        return getattr(logit_processor, attribute_name, None)
    return None