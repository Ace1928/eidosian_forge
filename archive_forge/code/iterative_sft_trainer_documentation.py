import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
from transformers.trainer_utils import EvalLoopOutput
from ..core import PPODecorators
from ..import_utils import is_peft_available

        Run an optimisation step given a list of input_ids, attention_mask, and labels or a list of text and text_labels.
        Args:
            input_ids (List[`torch.LongTensor`]):
                List of tensors containing the input_ids (if not provided, text will be used)
            attention_mask (List[`torch.LongTensor`], , *optional*):
                List of tensors containing the attention_mask
            labels (List[`torch.FloatTensor`], *optional*):
                List of tensors containing the labels (if set to None, will default to input_ids)
            texts (List[`str`], *optional*):
                List of strings containing the text input (if not provided, input_ids will directly be used)
            texts_labels (List[`str`], *optional*):
                List of strings containing the text labels (if set to None, will default to text)
        Returns:
            `dict[str, Any]`: A summary of the training statistics
        