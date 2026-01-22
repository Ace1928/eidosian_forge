import inspect
import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, Union, get_args
from .constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME, SAFETENSORS_SINGLE_FILE
from .file_download import hf_hub_download
from .hf_api import HfApi
from .repocard import ModelCard, ModelCardData
from .utils import (
from .utils._deprecation import _deprecate_arguments
def _load_dataclass(datacls: Type['DataclassInstance'], data: dict) -> 'DataclassInstance':
    """Load a dataclass instance from a dictionary.

    Fields not expected by the dataclass are ignored.
    """
    return datacls(**{k: v for k, v in data.items() if k in datacls.__dataclass_fields__})