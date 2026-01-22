import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _is_modin_series(data: DataType) -> bool:
    try:
        import modin.pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.Series)