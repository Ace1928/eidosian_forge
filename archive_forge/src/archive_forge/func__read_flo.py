import itertools
import os
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from ..io.image import _read_png_16
from .utils import _read_pfm, verify_str_arg
from .vision import VisionDataset
def _read_flo(file_name: str) -> np.ndarray:
    """Read .flo file in Middlebury format"""
    with open(file_name, 'rb') as f:
        magic = np.fromfile(f, 'c', count=4).tobytes()
        if magic != b'PIEH':
            raise ValueError('Magic number incorrect. Invalid .flo file')
        w = int(np.fromfile(f, '<i4', count=1))
        h = int(np.fromfile(f, '<i4', count=1))
        data = np.fromfile(f, '<f4', count=2 * w * h)
        return data.reshape(h, w, 2).transpose(2, 0, 1)