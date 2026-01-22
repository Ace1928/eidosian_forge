from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import torch
import bitsandbytes.functional as F
def _get_tile_size(format):
    assert format in ('col_turing', 'col_ampere'), f'please find this assert and manually enter tile size for {format}'
    return (8, 32) if format == 'col_turing' else (32, 32)