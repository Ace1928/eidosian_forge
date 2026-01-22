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
def _to_yaml_dict(self) -> dict:
    yaml_dict = {}
    dataset_info_dict = asdict(self)
    for key in dataset_info_dict:
        if key in self._INCLUDED_INFO_IN_YAML:
            value = getattr(self, key)
            if hasattr(value, '_to_yaml_list'):
                yaml_dict[key] = value._to_yaml_list()
            elif hasattr(value, '_to_yaml_string'):
                yaml_dict[key] = value._to_yaml_string()
            else:
                yaml_dict[key] = value
    return yaml_dict