import bz2
import contextlib
import gzip
import hashlib
import itertools
import lzma
import os
import os.path
import pathlib
import re
import sys
import tarfile
import urllib
import urllib.error
import urllib.request
import warnings
import zipfile
from typing import Any, Callable, Dict, IO, Iterable, Iterator, List, Optional, Tuple, TypeVar
from urllib.parse import urlparse
import numpy as np
import requests
import torch
from torch.utils.model_zoo import tqdm
from .._internally_replaced_utils import _download_file_from_remote_location, _is_remote_location_available
def _read_pfm(file_name: str, slice_channels: int=2) -> np.ndarray:
    """Read file in .pfm format. Might contain either 1 or 3 channels of data.

    Args:
        file_name (str): Path to the file.
        slice_channels (int): Number of channels to slice out of the file.
            Useful for reading different data formats stored in .pfm files: Optical Flows, Stereo Disparity Maps, etc.
    """
    with open(file_name, 'rb') as f:
        header = f.readline().rstrip()
        if header not in [b'PF', b'Pf']:
            raise ValueError('Invalid PFM file')
        dim_match = re.match(b'^(\\d+)\\s(\\d+)\\s$', f.readline())
        if not dim_match:
            raise Exception('Malformed PFM header.')
        w, h = (int(dim) for dim in dim_match.groups())
        scale = float(f.readline().rstrip())
        if scale < 0:
            endian = '<'
            scale = -scale
        else:
            endian = '>'
        data = np.fromfile(f, dtype=endian + 'f')
    pfm_channels = 3 if header == b'PF' else 1
    data = data.reshape(h, w, pfm_channels).transpose(2, 0, 1)
    data = np.flip(data, axis=1)
    data = data[:slice_channels, :, :]
    return data.astype(np.float32)