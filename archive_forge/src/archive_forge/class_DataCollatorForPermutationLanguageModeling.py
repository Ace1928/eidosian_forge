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
@dataclass
class DataCollatorForPermutationLanguageModeling(DataCollatorMixin):
    """
    Data collator used for permutation language modeling.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for permutation language modeling with procedures specific to XLNet
    """
    tokenizer: PreTrainedTokenizerBase
    plm_probability: float = 1 / 6
    max_span_length: int = 5
    return_tensors: str = 'pt'

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            examples = [e['input_ids'] for e in examples]
        batch = _torch_collate_batch(examples, self.tokenizer)
        inputs, perm_mask, target_mapping, labels = self.torch_mask_tokens(batch)
        return {'input_ids': inputs, 'perm_mask': perm_mask, 'target_mapping': target_mapping, 'labels': labels}

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            examples = [e['input_ids'] for e in examples]
        batch = _tf_collate_batch(examples, self.tokenizer)
        inputs, perm_mask, target_mapping, labels = self.tf_mask_tokens(batch)
        return {'input_ids': inputs, 'perm_mask': perm_mask, 'target_mapping': target_mapping, 'labels': labels}

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            examples = [e['input_ids'] for e in examples]
        batch = _numpy_collate_batch(examples, self.tokenizer)
        inputs, perm_mask, target_mapping, labels = self.numpy_mask_tokens(batch)
        return {'input_ids': inputs, 'perm_mask': perm_mask, 'target_mapping': target_mapping, 'labels': labels}

    def torch_mask_tokens(self, inputs: Any) -> Tuple[Any, Any, Any, Any]:
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
        import torch
        if self.tokenizer.mask_token is None:
            raise ValueError('This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer.')
        if inputs.size(1) % 2 != 0:
            raise ValueError('This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see relevant comments in source code for details.')
        labels = inputs.clone()
        masked_indices = torch.full(labels.shape, 0, dtype=torch.bool)
        target_mapping = torch.zeros((labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32)
        for i in range(labels.size(0)):
            cur_len = 0
            max_len = labels.size(1)
            while cur_len < max_len:
                span_length = torch.randint(1, self.max_span_length + 1, (1,)).item()
                context_length = int(span_length / self.plm_probability)
                start_index = cur_len + torch.randint(context_length - span_length + 1, (1,)).item()
                masked_indices[i, start_index:start_index + span_length] = 1
                cur_len += context_length
            target_mapping[i] = torch.eye(labels.size(1))
        special_tokens_mask = torch.tensor([self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()], dtype=torch.bool)
        masked_indices.masked_fill_(special_tokens_mask, value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            masked_indices.masked_fill_(padding_mask, value=0.0)
        non_func_mask = ~(padding_mask | special_tokens_mask)
        inputs[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100
        perm_mask = torch.zeros((labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32)
        for i in range(labels.size(0)):
            perm_index = torch.arange(labels.size(1))
            perm_index = perm_index.reshape((-1, labels.size(1) // 2)).transpose(0, 1)
            perm_index = perm_index[torch.randperm(labels.size(1) // 2)]
            perm_index = torch.flatten(perm_index.transpose(0, 1))
            perm_index.masked_fill_(~masked_indices[i] & non_func_mask[i], -1)
            perm_mask[i] = (perm_index.reshape((labels.size(1), 1)) <= perm_index.reshape((1, labels.size(1)))) & masked_indices[i]
        return (inputs.long(), perm_mask, target_mapping, labels.long())

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

    def numpy_mask_tokens(self, inputs: Any) -> Tuple[Any, Any, Any, Any]:
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
        if self.tokenizer.mask_token is None:
            raise ValueError('This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer.')
        if inputs.shape[1] % 2 != 0:
            raise ValueError('This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see relevant comments in source code for details.')
        labels = np.copy(inputs)
        masked_indices = np.full(labels.shape, 0, dtype=bool)
        target_mapping = np.zeros((labels.shape[0], labels.shape[1], labels.shape[1]), dtype=np.float32)
        for i in range(labels.shape[0]):
            cur_len = 0
            max_len = labels.shape[1]
            while cur_len < max_len:
                span_length = randint(1, self.max_span_length + 1)
                context_length = int(span_length / self.plm_probability)
                start_index = cur_len + randint(0, context_length - span_length + 1)
                masked_indices[i, start_index:start_index + span_length] = 1
                cur_len += context_length
            target_mapping[i] = np.eye(labels.shape[1])
        special_tokens_mask = np.array([self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()], dtype=bool)
        masked_indices[special_tokens_mask] = 0
        if self.tokenizer._pad_token is not None:
            padding_mask = labels == self.tokenizer.pad_token_id
            masked_indices[padding_mask] = 0.0
        non_func_mask = ~(padding_mask | special_tokens_mask)
        inputs[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100
        perm_mask = np.zeros((labels.shape[0], labels.shape[1], labels.shape[1]), dtype=np.float32)
        for i in range(labels.shape[0]):
            perm_index = np.arange(labels.shape[1])
            perm_index = perm_index.reshape((-1, labels.shape[1] // 2)).T
            np.random.shuffle(perm_index)
            perm_index = perm_index.T.flatten()
            perm_index[~masked_indices[i] & non_func_mask[i]] = -1
            perm_mask[i] = (perm_index.reshape((labels.shape[1], 1)) <= perm_index.reshape((1, labels.shape[1]))) & masked_indices[i]
        return (inputs.astype(np.int64), perm_mask, target_mapping, labels.astype(np.int64))