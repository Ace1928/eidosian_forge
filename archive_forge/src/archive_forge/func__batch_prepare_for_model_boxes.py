import os
import re
import warnings
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, logging
@add_end_docstrings(UDOP_ENCODE_KWARGS_DOCSTRING)
def _batch_prepare_for_model_boxes(self, batch_text_or_text_pairs, is_pair: bool=None, boxes: Optional[List[List[int]]]=None, word_labels: Optional[List[List[int]]]=None, add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[str]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_length: bool=False, verbose: bool=True) -> BatchEncoding:
    """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """
    batch_outputs = {}
    for idx, example in enumerate(zip(batch_text_or_text_pairs, boxes)):
        batch_text_or_text_pair, boxes_example = example
        outputs = self.prepare_for_model_boxes(batch_text_or_text_pair[0] if is_pair else batch_text_or_text_pair, batch_text_or_text_pair[1] if is_pair else None, boxes_example, word_labels=word_labels[idx] if word_labels is not None else None, add_special_tokens=add_special_tokens, padding=PaddingStrategy.DO_NOT_PAD.value, truncation=truncation_strategy.value, max_length=max_length, stride=stride, pad_to_multiple_of=None, return_attention_mask=False, return_token_type_ids=return_token_type_ids, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, return_tensors=None, prepend_batch_axis=False, verbose=verbose)
        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)
    batch_outputs = self.pad(batch_outputs, padding=padding_strategy.value, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
    batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
    return batch_outputs