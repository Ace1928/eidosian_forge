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
class ORTModelForImageClassification(ORTModel):
    """
    ONNX Model for image-classification tasks. This class officially supports beit, convnext, convnextv2, data2vec_vision, deit, levit, mobilenet_v1, mobilenet_v2, mobilevit, poolformer, resnet, segformer, swin, vit.
    """
    auto_model_class = AutoModelForImageClassification

    @add_start_docstrings_to_model_forward(ONNX_IMAGE_INPUTS_DOCSTRING.format('batch_size, num_channels, height, width') + IMAGE_CLASSIFICATION_EXAMPLE.format(processor_class=_FEATURE_EXTRACTOR_FOR_DOC, model_class='ORTModelForImageClassification', checkpoint='optimum/vit-base-patch16-224'))
    def forward(self, pixel_values: Union[torch.Tensor, np.ndarray], **kwargs):
        use_torch = isinstance(pixel_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        if self.device.type == 'cuda' and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(pixel_values, ordered_input_names=self._ordered_input_names)
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            return ImageClassifierOutput(logits=output_buffers['logits'].view(output_shapes['logits']))
        else:
            if use_torch:
                pixel_values = pixel_values.cpu().detach().numpy()
            onnx_inputs = {'pixel_values': pixel_values}
            outputs = self.model.run(None, onnx_inputs)
            logits = outputs[self.output_names['logits']]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)
            return ImageClassifierOutput(logits=logits)