import math
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _input_data_collator(batch: Dict[str, Tensor], device: Optional[Union[str, torch.device]]=None) -> Dict[str, Tensor]:
    """Trim model inputs.

    This function trims the model inputs to the longest sequence within the batch and put the input on the proper
    device.

    """
    max_len = int(batch['attention_mask'].sum(1).max().item())
    input_ids = batch['input_ids'][:, :max_len].to(device)
    attention_mask = batch['attention_mask'][:, :max_len].to(device)
    batch.update({'input_ids': input_ids, 'attention_mask': attention_mask})
    return batch