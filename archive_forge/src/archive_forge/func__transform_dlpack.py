import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _transform_dlpack(data: DataType) -> bool:
    from cupy import fromDlpack
    assert 'used_dltensor' not in str(data)
    data = fromDlpack(data)
    return data