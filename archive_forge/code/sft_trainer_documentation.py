import dataclasses
import inspect
import warnings
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from accelerate.state import PartialState
from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from transformers import (
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from ..extras.dataset_formatting import get_formatting_func_from_dataset
from ..import_utils import is_peft_available
from .utils import (

        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        Since in transformers Trainer we do have an `_activate_neftune` method, we need to rename this method to avoid conflicts.
        