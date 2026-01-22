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
def _infolm_compute(model: 'PreTrainedModel', preds_dataloader: DataLoader, target_dataloader: DataLoader, temperature: float, idf: bool, information_measure_cls: _InformationMeasure, special_tokens_map: Dict[str, int], verbose: bool=True) -> Tensor:
    """Calculate selected information measure using the pre-trained language model.

    Args:
        model:
            Initialized model from HuggingFace's `transformers package.
        preds_dataloader:
            Loader iterating over tokenizer predicted sentences.
        target_dataloader:
            Loader iterating over tokenizer reference sentences.
        temperature:
            A temperature for calibrating language modelling. For more information, please reference `InfoLM`_ paper.
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
        information_measure_cls:
            Information measure class containing all parameters necessary for calculating information measure values
            using ``preds_distribution`` and ``target_distribution``.
        special_tokens_map:
            A dictionary mapping tokenizer special tokens into the corresponding integer values.
        verbose:
            An indication of whether a progress bar to be displayed during the embeddings calculation.

    Return:
        A corpus-level InfoLM score.

    """
    preds_distribution = _get_data_distribution(model, preds_dataloader, temperature, idf, special_tokens_map, verbose)
    target_distribution = _get_data_distribution(model, target_dataloader, temperature, idf, special_tokens_map, verbose)
    preds_distribution = preds_distribution[preds_dataloader.dataset.sorting_indices]
    target_distribution = target_distribution[target_dataloader.dataset.sorting_indices]
    return information_measure_cls(preds_distribution, target_distribution)