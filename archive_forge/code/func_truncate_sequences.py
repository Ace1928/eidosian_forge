import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece as spm
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import (
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, logging
from ..xlm_roberta.tokenization_xlm_roberta import (
def truncate_sequences(self, ids: List[int], token_boxes: List[List[int]], pair_ids: Optional[List[int]]=None, pair_token_boxes: Optional[List[List[int]]]=None, labels: Optional[List[int]]=None, num_tokens_to_remove: int=0, truncation_strategy: Union[str, TruncationStrategy]='longest_first', stride: int=0) -> Tuple[List[int], List[int], List[int]]:
    """
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            token_boxes (`List[List[int]]`):
                Bounding boxes of the first sequence.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
            pair_token_boxes (`List[List[int]]`, *optional*):
                Bounding boxes of the second sequence.
            labels (`List[int]`, *optional*):
                Labels of the first sequence (for token classification tasks).
            num_tokens_to_remove (`int`, *optional*, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (`str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
                The strategy to follow for truncation. Can be:

                - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will truncate
                  token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a
                  batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater
                  than the model maximum admissible input size).
            stride (`int`, *optional*, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            `Tuple[List[int], List[int], List[int]]`: The truncated `ids`, the truncated `pair_ids` and the list of
            overflowing tokens.
        """
    if num_tokens_to_remove <= 0:
        return (ids, token_boxes, pair_ids, pair_token_boxes, labels, [], [], [])
    if not isinstance(truncation_strategy, TruncationStrategy):
        truncation_strategy = TruncationStrategy(truncation_strategy)
    overflowing_tokens = []
    overflowing_token_boxes = []
    overflowing_labels = []
    if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
        for _ in range(num_tokens_to_remove):
            if pair_ids is None or len(ids) > len(pair_ids):
                if not overflowing_tokens:
                    window_len = min(len(ids), stride + 1)
                else:
                    window_len = 1
                overflowing_tokens.extend(ids[-window_len:])
                overflowing_token_boxes.extend(token_boxes[-window_len:])
                overflowing_labels.extend(labels[-window_len:])
                ids = ids[:-1]
                token_boxes = token_boxes[:-1]
                labels = labels[:-1]
            else:
                if not overflowing_tokens:
                    window_len = min(len(pair_ids), stride + 1)
                else:
                    window_len = 1
                overflowing_tokens.extend(pair_ids[-window_len:])
                overflowing_token_boxes.extend(pair_token_boxes[-window_len:])
                pair_ids = pair_ids[:-1]
                pair_token_boxes = pair_token_boxes[:-1]
    elif truncation_strategy == TruncationStrategy.ONLY_FIRST:
        if len(ids) > num_tokens_to_remove:
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            overflowing_token_boxes = token_boxes[-window_len:]
            overflowing_labels = labels[-window_len:]
            ids = ids[:-num_tokens_to_remove]
            token_boxes = token_boxes[:-num_tokens_to_remove]
            labels = labels[:-num_tokens_to_remove]
        else:
            logger.error(f"We need to remove {num_tokens_to_remove} to truncate the input but the first sequence has a length {len(ids)}. Please select another truncation strategy than {truncation_strategy}, for instance 'longest_first' or 'only_second'.")
    elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
        if len(pair_ids) > num_tokens_to_remove:
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            overflowing_token_boxes = pair_token_boxes[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
            pair_token_boxes = pair_token_boxes[:-num_tokens_to_remove]
        else:
            logger.error(f"We need to remove {num_tokens_to_remove} to truncate the input but the second sequence has a length {len(pair_ids)}. Please select another truncation strategy than {truncation_strategy}, for instance 'longest_first' or 'only_first'.")
    return (ids, token_boxes, pair_ids, pair_token_boxes, labels, overflowing_tokens, overflowing_token_boxes, overflowing_labels)