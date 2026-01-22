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
@classmethod
def _from_yaml_list(cls, yaml_data: list) -> 'Features':
    yaml_data = copy.deepcopy(yaml_data)

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

    def from_yaml_inner(obj: Union[dict, list]) -> Union[dict, list]:
        if isinstance(obj, dict):
            if not obj:
                return {}
            _type = next(iter(obj))
            if _type == 'sequence':
                _feature = unsimplify(obj).pop(_type)
                return {'feature': from_yaml_inner(_feature), **obj, '_type': 'Sequence'}
            if _type == 'list':
                return [from_yaml_inner(unsimplify(obj)[_type])]
            if _type == 'struct':
                return from_yaml_inner(obj['struct'])
            elif _type == 'dtype':
                if isinstance(obj['dtype'], str):
                    try:
                        Value(obj['dtype'])
                        return {**obj, '_type': 'Value'}
                    except ValueError:
                        return {'_type': snakecase_to_camelcase(obj['dtype'])}
                else:
                    return from_yaml_inner(obj['dtype'])
            else:
                return {'_type': snakecase_to_camelcase(_type), **unsimplify(obj)[_type]}
        elif isinstance(obj, list):
            names = [_feature.pop('name') for _feature in obj]
            return {name: from_yaml_inner(_feature) for name, _feature in zip(names, obj)}
        else:
            raise TypeError(f'Expected a dict or a list but got {type(obj)}: {obj}')
    return cls.from_dict(from_yaml_inner(yaml_data))