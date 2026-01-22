import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5HifiGan
from transformers.utils import is_tf_available, is_torch_available
from ...utils import (
from ...utils.import_utils import _diffusers_version
from ..tasks import TasksManager
from .constants import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_ENCODER_NAME
def _get_submodels_for_export_encoder_decoder(model: Union['PreTrainedModel', 'TFPreTrainedModel'], use_past: bool) -> Dict[str, Union['PreTrainedModel', 'TFPreTrainedModel']]:
    """
    Returns the encoder and decoder parts of the model.
    """
    models_for_export = {}
    encoder_model = model.get_encoder()
    models_for_export[ONNX_ENCODER_NAME] = encoder_model
    models_for_export[ONNX_DECODER_NAME] = model
    if use_past:
        models_for_export[ONNX_DECODER_WITH_PAST_NAME] = model
    return models_for_export