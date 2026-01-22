import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, FastSpeech2ConformerEncoder):
        module.gradient_checkpointing = value