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
def _strval2int(self, value: str) -> int:
    failed_parse = False
    value = str(value)
    int_value = self._str2int.get(value)
    if int_value is None:
        int_value = self._str2int.get(value.strip())
        if int_value is None:
            try:
                int_value = int(value)
            except ValueError:
                failed_parse = True
            else:
                if int_value < -1 or int_value >= self.num_classes:
                    failed_parse = True
    if failed_parse:
        raise ValueError(f'Invalid string class label {value}')
    return int_value