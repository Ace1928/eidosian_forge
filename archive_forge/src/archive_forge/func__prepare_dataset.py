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
def _prepare_dataset(self, dataset, tokenizer, packing, dataset_text_field, max_seq_length, formatting_func, num_of_sequences, chars_per_token, remove_unused_columns=True, append_concat_token=True, add_special_tokens=True):
    if dataset is None:
        raise ValueError('The dataset should not be None')
    if isinstance(dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset, ConstantLengthDataset)):
        return dataset
    if not packing:
        return self._prepare_non_packed_dataloader(tokenizer, dataset, dataset_text_field, max_seq_length, formatting_func, add_special_tokens, remove_unused_columns)
    else:
        return self._prepare_packed_dataloader(tokenizer, dataset, dataset_text_field, max_seq_length, num_of_sequences, chars_per_token, formatting_func, append_concat_token, add_special_tokens)