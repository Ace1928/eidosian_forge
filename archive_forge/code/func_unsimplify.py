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
def unsimplify(feature: dict) -> dict:
    if not isinstance(feature, dict):
        raise TypeError(f'Expected a dict but got a {type(feature)}: {feature}')
    if isinstance(feature.get('sequence'), str):
        feature['sequence'] = {'dtype': feature['sequence']}
    if isinstance(feature.get('list'), str):
        feature['list'] = {'dtype': feature['list']}
    if isinstance(feature.get('class_label'), dict) and isinstance(feature['class_label'].get('names'), dict):
        label_ids = sorted(feature['class_label']['names'], key=int)
        if label_ids and [int(label_id) for label_id in label_ids] != list(range(int(label_ids[-1]) + 1)):
            raise ValueError(f'ClassLabel expected a value for all label ids [0:{int(label_ids[-1]) + 1}] but some ids are missing.')
        feature['class_label']['names'] = [feature['class_label']['names'][label_id] for label_id in label_ids]
    return feature