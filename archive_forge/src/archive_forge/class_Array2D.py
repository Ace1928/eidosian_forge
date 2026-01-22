import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import InitVar, dataclass, field, fields
from functools import reduce, wraps
from operator import mul
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from typing import Sequence as Sequence_
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
import pyarrow_hotfix  # noqa: F401  # to fix vulnerability on pyarrow<14.0.1
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype
from .. import config
from ..naming import camelcase_to_snakecase, snakecase_to_camelcase
from ..table import array_cast
from ..utils import logging
from ..utils.py_utils import asdict, first_non_null_value, zip_dict
from .audio import Audio
from .image import Image, encode_pil_image
from .translation import Translation, TranslationVariableLanguages
@dataclass
class Array2D(_ArrayXD):
    """Create a two-dimensional array.

    Args:
        shape (`tuple`):
            The size of each dimension.
        dtype (`str`):
            The value of the data type.

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'x': Array2D(shape=(1, 3), dtype='int32')})
    ```
    """
    shape: tuple
    dtype: str
    id: Optional[str] = None
    _type: str = field(default='Array2D', init=False, repr=False)