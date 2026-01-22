import logging
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import (
from transformers.file_utils import add_end_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
import onnxruntime as ort
from ..exporters import TasksManager
from ..exporters.onnx import main_export
from ..modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from ..onnx.utils import _get_external_data_paths
from ..utils.file_utils import find_files_matching_pattern
from ..utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from .io_binding import IOBindingHelper, TypeHelper
from .utils import (
@staticmethod
def _cached_file(model_path: Union[Path, str], use_auth_token: Optional[Union[bool, str]]=None, revision: Optional[str]=None, force_download: bool=False, cache_dir: Optional[str]=None, file_name: Optional[str]=None, subfolder: str='', local_files_only: bool=False):
    model_path = Path(model_path)
    if model_path.is_dir():
        model_cache_path = model_path / file_name
        preprocessors = maybe_load_preprocessors(model_path.as_posix())
    else:
        model_cache_path = hf_hub_download(repo_id=model_path.as_posix(), filename=file_name, subfolder=subfolder, use_auth_token=use_auth_token, revision=revision, cache_dir=cache_dir, force_download=force_download, local_files_only=local_files_only)
        try:
            hf_hub_download(repo_id=model_path.as_posix(), subfolder=subfolder, filename=file_name + '_data', use_auth_token=use_auth_token, revision=revision, cache_dir=cache_dir, force_download=force_download, local_files_only=local_files_only)
        except EntryNotFoundError:
            pass
        model_cache_path = Path(model_cache_path)
        preprocessors = maybe_load_preprocessors(model_path.as_posix(), subfolder=subfolder)
    return (model_cache_path, preprocessors)