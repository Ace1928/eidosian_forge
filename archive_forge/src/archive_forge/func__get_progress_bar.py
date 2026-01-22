import math
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _get_progress_bar(dataloader: DataLoader, verbose: bool=False) -> Union[DataLoader, 'tqdm.auto.tqdm']:
    """Wrap dataloader in progressbar if asked for.

    Function will return either the dataloader itself when `verbose = False`, or it wraps the dataloader with
    `tqdm.auto.tqdm`, when `verbose = True` to display a progress bar during the embbeddings calculation.

    """
    import tqdm
    return tqdm.auto.tqdm(dataloader) if verbose else dataloader