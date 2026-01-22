import os
from enum import unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from typing_extensions import Literal
from torchmetrics.functional.text.helper_embedding_metric import (
from torchmetrics.utilities.enums import EnumStr
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_4
def _get_batch_distribution(model: 'PreTrainedModel', batch: Dict[str, Tensor], temperature: float, idf: bool, special_tokens_map: Dict[str, int]) -> Tensor:
    """Calculate a discrete probability distribution for a batch of examples. See `InfoLM`_ for details.

    Args:
        model:
            Initialized model from HuggingFace's `transformers package.
        batch:
            An input batch dictionary containing ``input_ids`` and ``attention_mask``.
        temperature:
            A temperature for calibrating language modelling. For more information, please reference `InfoLM`_ paper.
        max_length:
            A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
        special_tokens_map:
            A dictionary mapping tokenizer special tokens into the corresponding integer values.

    Return:
        A discrete probability distribution.

    """
    seq_len = batch['input_ids'].shape[1]
    prob_distribution_batch_list: List[Tensor] = []
    token_mask = _get_token_mask(batch['input_ids'], special_tokens_map['pad_token_id'], special_tokens_map['sep_token_id'], special_tokens_map['cls_token_id'])
    for mask_idx in range(seq_len):
        input_ids = batch['input_ids'].clone()
        input_ids[:, mask_idx] = special_tokens_map['mask_token_id']
        logits_distribution = model(input_ids, batch['attention_mask']).logits
        logits_distribution = logits_distribution[:, mask_idx, :]
        prob_distribution = F.softmax(logits_distribution / temperature, dim=-1)
        if idf:
            prob_distribution *= batch['input_ids_idf'][:, mask_idx].unsqueeze(1).to(prob_distribution.device)
        prob_distribution_batch_list.append(prob_distribution.unsqueeze(1).cpu())
        del input_ids, logits_distribution, prob_distribution
    prob_distribution_batch = torch.cat(prob_distribution_batch_list, dim=1)
    prob_distribution_batch = torch.einsum('bsv, bs -> bsv', prob_distribution_batch.to(token_mask.device), token_mask)
    if idf:
        masked_input_ids_idf = token_mask * batch['input_ids_idf'].to(token_mask.device)
        return prob_distribution_batch.sum(dim=1) / masked_input_ids_idf.sum(dim=1).unsqueeze(1)
    return prob_distribution_batch.sum(dim=1) / token_mask.sum(dim=1).unsqueeze(1)