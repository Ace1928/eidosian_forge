import copy
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests
import yaml
from huggingface_hub import model_info
from huggingface_hub.utils import HFValidationError
from . import __version__
from .models.auto.modeling_auto import (
from .training_args import ParallelMode
from .utils import (
def _get_mapping_values(mapping):
    result = []
    for v in mapping.values():
        if isinstance(v, (tuple, list)):
            result += list(v)
        else:
            result.append(v)
    return result