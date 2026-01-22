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
def _align_features(features_list: List[Features]) -> List[Features]:
    """Align dictionaries of features so that the keys that are found in multiple dictionaries share the same feature."""
    name2feature = {}
    for features in features_list:
        for k, v in features.items():
            if k in name2feature and isinstance(v, dict):
                name2feature[k] = _align_features([name2feature[k], v])[0]
            elif k not in name2feature or (isinstance(name2feature[k], Value) and name2feature[k].dtype == 'null'):
                name2feature[k] = v
    return [Features({k: name2feature[k] for k in features.keys()}) for features in features_list]