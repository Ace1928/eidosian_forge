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
def to_yaml_inner(obj: Union[dict, list]) -> dict:
    if isinstance(obj, dict):
        _type = obj.pop('_type', None)
        if _type == 'Sequence':
            _feature = obj.pop('feature')
            return simplify({'sequence': to_yaml_inner(_feature), **obj})
        elif _type == 'Value':
            return obj
        elif _type and (not obj):
            return {'dtype': camelcase_to_snakecase(_type)}
        elif _type:
            return {'dtype': simplify({camelcase_to_snakecase(_type): obj})}
        else:
            return {'struct': [{'name': name, **to_yaml_inner(_feature)} for name, _feature in obj.items()]}
    elif isinstance(obj, list):
        return simplify({'list': simplify(to_yaml_inner(obj[0]))})
    elif isinstance(obj, tuple):
        return to_yaml_inner(list(obj))
    else:
        raise TypeError(f'Expected a dict or a list but got {type(obj)}: {obj}')