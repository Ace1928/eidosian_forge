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
class _ArrayXDExtensionType(pa.ExtensionType):
    ndims: Optional[int] = None

    def __init__(self, shape: tuple, dtype: str):
        if self.ndims is None or self.ndims <= 1:
            raise ValueError('You must instantiate an array type with a value for dim that is > 1')
        if len(shape) != self.ndims:
            raise ValueError(f"shape={shape} and ndims={self.ndims} don't match")
        for dim in range(1, self.ndims):
            if shape[dim] is None:
                raise ValueError(f'Support only dynamic size on first dimension. Got: {shape}')
        self.shape = tuple(shape)
        self.value_type = dtype
        self.storage_dtype = self._generate_dtype(self.value_type)
        pa.ExtensionType.__init__(self, self.storage_dtype, f'{self.__class__.__module__}.{self.__class__.__name__}')

    def __arrow_ext_serialize__(self):
        return json.dumps((self.shape, self.value_type)).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        args = json.loads(serialized)
        return cls(*args)

    def __reduce__(self):
        return (self.__arrow_ext_deserialize__, (self.storage_type, self.__arrow_ext_serialize__()))

    def __hash__(self):
        return hash((self.__class__, self.shape, self.value_type))

    def __arrow_ext_class__(self):
        return ArrayExtensionArray

    def _generate_dtype(self, dtype):
        dtype = string_to_arrow(dtype)
        for d in reversed(self.shape):
            dtype = pa.list_(dtype)
        return dtype

    def to_pandas_dtype(self):
        return PandasArrayExtensionDtype(self.value_type)