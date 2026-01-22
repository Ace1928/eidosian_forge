import math
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
class TokenizedDataset(TextDataset):
    """The child class of `TextDataset` class used with already tokenized data."""

    def __init__(self, input_ids: Tensor, attention_mask: Tensor, idf: bool=False, tokens_idf: Optional[Dict[int, float]]=None) -> None:
        """Initialize the dataset class.

        Args:
            input_ids: Input indexes
            attention_mask: Attention mask
            idf:
                An indication of whether calculate token inverse document frequencies to weight the model embeddings.
            tokens_idf: Inverse document frequencies (these should be calculated on reference sentences).

        """
        text = dict(zip(['input_ids', 'attention_mask', 'sorting_indices'], _sort_data_according_length(input_ids, attention_mask)))
        self.sorting_indices = text.pop('sorting_indices')
        self.text = _input_data_collator(text)
        self.num_sentences = len(self.text['input_ids'])
        self.max_length = self.text['input_ids'].shape[1]
        self.idf = idf
        self.tokens_idf = {}
        if idf:
            self.tokens_idf = tokens_idf if tokens_idf is not None else self._get_tokens_idf()