import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _is_arrow(data: DataType) -> bool:
    try:
        import pyarrow as pa
        from pyarrow import dataset as arrow_dataset
        return isinstance(data, (pa.Table, arrow_dataset.Dataset))
    except ImportError:
        return False