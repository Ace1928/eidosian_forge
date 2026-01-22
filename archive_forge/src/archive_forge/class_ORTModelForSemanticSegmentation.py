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
@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForSemanticSegmentation(ORTModel):
    """
    ONNX Model for semantic-segmentation, with an all-MLP decode head on top e.g. for ADE20k, CityScapes. This class officially supports segformer.
    """
    auto_model_class = AutoModelForSemanticSegmentation

    @add_start_docstrings_to_model_forward(ONNX_IMAGE_INPUTS_DOCSTRING.format('batch_size, num_channels, height, width') + SEMANTIC_SEGMENTATION_EXAMPLE.format(processor_class=_FEATURE_EXTRACTOR_FOR_DOC, model_class='ORTModelForSemanticSegmentation', checkpoint='optimum/segformer-b0-finetuned-ade-512-512'))
    def forward(self, **kwargs):
        use_torch = isinstance(next(iter(kwargs.values())), torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        if self.device.type == 'cuda' and self.use_io_binding:
            io_binding = IOBindingHelper.prepare_io_binding(self, **kwargs, ordered_input_names=self._ordered_input_names)
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            outputs = {}
            for name, output in zip(self.output_names.keys(), io_binding._iobinding.get_outputs()):
                outputs[name] = IOBindingHelper.to_pytorch(output)
            return SemanticSegmenterOutput(logits=outputs['logits'])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch=use_torch, **kwargs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            logits = onnx_outputs[self.output_names['logits']]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)
            return SemanticSegmenterOutput(logits=logits)

    def _prepare_onnx_inputs(self, use_torch: bool, **kwargs):
        onnx_inputs = {}
        for input in self.inputs_names.keys():
            onnx_inputs[input] = kwargs.pop(input)
            if use_torch:
                onnx_inputs[input] = onnx_inputs[input].cpu().detach().numpy()
        return onnx_inputs