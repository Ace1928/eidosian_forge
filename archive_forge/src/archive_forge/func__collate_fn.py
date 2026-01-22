import bisect
import math
import warnings
from fractions import Fraction
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import torch
from torchvision.io import _probe_video_from_file, _read_video_from_file, read_video, read_video_timestamps
from .utils import tqdm
def _collate_fn(x: T) -> T:
    """
    Dummy collate function to be used with _VideoTimestampsDataset
    """
    return x