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
@dataclass
class PostProcessedInfo:
    features: Optional[Features] = None
    resources_checksums: Optional[dict] = None

    def __post_init__(self):
        if self.features is not None and (not isinstance(self.features, Features)):
            self.features = Features.from_dict(self.features)

    @classmethod
    def from_dict(cls, post_processed_info_dict: dict) -> 'PostProcessedInfo':
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in post_processed_info_dict.items() if k in field_names})