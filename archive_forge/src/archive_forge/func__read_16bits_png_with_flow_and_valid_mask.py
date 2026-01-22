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
def _read_16bits_png_with_flow_and_valid_mask(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    flow_and_valid = _read_png_16(file_name).to(torch.float32)
    flow, valid_flow_mask = (flow_and_valid[:2, :, :], flow_and_valid[2, :, :])
    flow = (flow - 2 ** 15) / 64
    valid_flow_mask = valid_flow_mask.bool()
    return (flow.numpy(), valid_flow_mask.numpy())