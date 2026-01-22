import os.path
from typing import Any, Callable, cast, Optional, Tuple
import numpy as np
from PIL import Image
from .utils import check_integrity, download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
def _verify_folds(self, folds: Optional[int]) -> Optional[int]:
    if folds is None:
        return folds
    elif isinstance(folds, int):
        if folds in range(10):
            return folds
        msg = 'Value for argument folds should be in the range [0, 10), but got {}.'
        raise ValueError(msg.format(folds))
    else:
        msg = 'Expected type None or int for argument folds, but got type {}.'
        raise ValueError(msg.format(type(folds)))