import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type, TypeVar, Union, get_args
from .constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME, SAFETENSORS_SINGLE_FILE
from .file_download import hf_hub_download
from .hf_api import HfApi
from .utils import (
from .utils._deprecation import _deprecate_arguments
Load Pytorch pretrained weights and return the loaded model.