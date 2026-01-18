import collections
import contextlib
import dataclasses
import os
import shutil
import tempfile
import textwrap
import time
from typing import cast, Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple
import uuid
import torch
def trim_sigfig(x: float, n: int) -> float:
    """Trim `x` to `n` significant figures. (e.g. 3.14159, 2 -> 3.10000)"""
    assert n == int(n)
    magnitude = int(torch.tensor(x).abs().log10().ceil().item())
    scale = 10 ** (magnitude - n)
    return float(torch.tensor(x / scale).round() * scale)