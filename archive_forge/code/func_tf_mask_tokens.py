import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from ..models.bert import BertTokenizer, BertTokenizerFast
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import PaddingStrategy
def tf_mask_tokens(self, inputs: Any) -> Tuple[Any, Any, Any, Any]:
    """
        The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

            0. Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
            1. Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
            2. Reserve a context of length `context_length = span_length / plm_probability` to surround span to be
               masked
            3. Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length -
               span_length]` and mask tokens `start_index:start_index + span_length`
            4. Set `cur_len = cur_len + context_length`. If `cur_len < max_len` (i.e. there are tokens remaining in the
               sequence to be processed), repeat from Step 1.
        """
    import tensorflow as tf
    if self.tokenizer.mask_token is None:
        raise ValueError('This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer.')
    if tf.shape(inputs)[1] % 2 != 0:
        raise ValueError('This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see relevant comments in source code for details.')
    labels = tf.identity(inputs)
    masked_indices = np.full(labels.shape.as_list(), 0, dtype=bool)
    labels_shape = tf.shape(labels)
    target_mapping = np.zeros((labels_shape[0], labels_shape[1], labels_shape[1]), dtype=np.float32)
    for i in range(len(labels)):
        cur_len = 0
        max_len = tf.shape(labels)[1]
        while cur_len < max_len:
            span_length = randint(1, self.max_span_length + 1)
            context_length = int(span_length / self.plm_probability)
            start_index = cur_len + randint(0, context_length - span_length + 1)
            masked_indices[i, start_index:start_index + span_length] = 1
            cur_len += context_length
        target_mapping[i] = np.eye(labels_shape[1])
    masked_indices = tf.cast(tf.convert_to_tensor(masked_indices), dtype=tf.bool)
    target_mapping = tf.convert_to_tensor(target_mapping)
    special_tokens_mask = tf.convert_to_tensor([self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.numpy().tolist()])
    special_tokens_mask = tf.cast(special_tokens_mask, dtype=tf.bool)
    masked_indices = masked_indices & ~special_tokens_mask
    if self.tokenizer._pad_token is not None:
        padding_mask = labels == self.tokenizer.pad_token_id
        masked_indices = masked_indices & ~padding_mask
    non_func_mask = ~(padding_mask | special_tokens_mask)
    inputs = tf.where(masked_indices, self.tokenizer.mask_token_id, inputs)
    labels = tf.where(masked_indices, labels, -100)
    perm_mask = []
    for i in range(len(labels)):
        perm_index = tf.range(labels_shape[1])
        perm_index = tf.transpose(tf.reshape(perm_index, (-1, labels_shape[1] // 2)))
        perm_index = tf.random.shuffle(perm_index)
        perm_index = tf.reshape(tf.transpose(perm_index), (-1,))
        perm_index = tf.where(~masked_indices[i] & non_func_mask[i], -1, perm_index)
        perm_mask.append((tf.reshape(perm_index, (labels_shape[1], 1)) <= tf.reshape(perm_index, (1, labels_shape[1]))) & masked_indices[i])
    perm_mask = tf.stack(perm_mask, axis=0)
    return (tf.cast(inputs, tf.int64), tf.cast(perm_mask, tf.float32), target_mapping, tf.cast(labels, tf.int64))