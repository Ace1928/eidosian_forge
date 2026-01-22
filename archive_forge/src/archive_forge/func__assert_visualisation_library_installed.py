from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from fairscale.nn import FullyShardedDataParallel
def _assert_visualisation_library_installed() -> None:
    try:
        import PIL
        import matplotlib
    except ImportError:
        install_matplotlib = 'pip install matplotlib'
        install_pil = 'pip install Pillow'
        error_message = 'Visualizing memory plots requires matplotlib and Pillow installed'
        assert False, f'{error_message}: {install_matplotlib}, {install_pil}'