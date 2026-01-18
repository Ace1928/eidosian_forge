import bisect
import math
import warnings
from fractions import Fraction
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import torch
from torchvision.io import _probe_video_from_file, _read_video_from_file, read_video, read_video_timestamps
from .utils import tqdm

        Gets a subclip from a list of videos.

        Args:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        