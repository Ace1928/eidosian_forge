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
def _get_special_tokens_map(tokenizer: 'PreTrainedTokenizerBase') -> Dict[str, int]:
    """Build a dictionary of model/tokenizer special tokens.

    Args:
        tokenizer:
            Initialized tokenizer from HuggingFace's `transformers package.

    Return:
        A dictionary containing: mask_token_id, pad_token_id, sep_token_id and cls_token_id.

    """
    return {'mask_token_id': tokenizer.mask_token_id, 'pad_token_id': tokenizer.pad_token_id, 'sep_token_id': tokenizer.sep_token_id, 'cls_token_id': tokenizer.cls_token_id}