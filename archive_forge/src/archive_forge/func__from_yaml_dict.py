import copy
import dataclasses
import json
import os
import posixpath
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union
import fsspec
from huggingface_hub import DatasetCard, DatasetCardData
from . import config
from .features import Features, Value
from .splits import SplitDict
from .tasks import TaskTemplate, task_template_from_dict
from .utils import Version
from .utils.logging import get_logger
from .utils.py_utils import asdict, unique_values
@classmethod
def _from_yaml_dict(cls, yaml_data: dict) -> 'DatasetInfo':
    yaml_data = copy.deepcopy(yaml_data)
    if yaml_data.get('features') is not None:
        yaml_data['features'] = Features._from_yaml_list(yaml_data['features'])
    if yaml_data.get('splits') is not None:
        yaml_data['splits'] = SplitDict._from_yaml_list(yaml_data['splits'])
    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in yaml_data.items() if k in field_names})