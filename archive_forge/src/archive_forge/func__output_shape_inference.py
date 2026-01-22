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
def _output_shape_inference(self, axis_name: Union[str, int], dimensions: Dict[str, int]) -> Union[str, int]:
    """
        Infers the output shape of a given dynamic axis by using the `dimensions` mapping.

        For instance, for the following inputs:
            axis_name = "past_sequence_length + sequence_length"
            dimensions = {"batch_size": 2, "sequence_length": 3, "past_sequence_length": 7}

        The inferred shape is 3 + 7 = 10.
        """
    if isinstance(axis_name, int):
        return axis_name
    elif axis_name in dimensions:
        return dimensions[axis_name]
    tokens = []
    for idx, match_ in enumerate(re.finditer(self.output_shape_inference_pattern, axis_name)):
        groups = match_.groups()
        matched_group = None
        for idx, group in enumerate(groups):
            if group is not None:
                matched_group = idx
                break
        if matched_group == 0:
            dim = dimensions.get(groups[0], None)
            if dim is None or not isinstance(dim, int):
                return axis_name
            tokens.append(str(dim))
        else:
            tokens.append(groups[matched_group])
    return int(eval(' '.join(tokens)))