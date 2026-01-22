from typing import Optional, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mobilenet_v1 import MobileNetV1Config
def _build_tf_to_pytorch_map(model, config, tf_weights=None):
    """
    A map of modules from TF to PyTorch.
    """
    tf_to_pt_map = {}
    if isinstance(model, MobileNetV1ForImageClassification):
        backbone = model.mobilenet_v1
    else:
        backbone = model
    prefix = 'MobilenetV1/Conv2d_0/'
    tf_to_pt_map[prefix + 'weights'] = backbone.conv_stem.convolution.weight
    tf_to_pt_map[prefix + 'BatchNorm/beta'] = backbone.conv_stem.normalization.bias
    tf_to_pt_map[prefix + 'BatchNorm/gamma'] = backbone.conv_stem.normalization.weight
    tf_to_pt_map[prefix + 'BatchNorm/moving_mean'] = backbone.conv_stem.normalization.running_mean
    tf_to_pt_map[prefix + 'BatchNorm/moving_variance'] = backbone.conv_stem.normalization.running_var
    for i in range(13):
        tf_index = i + 1
        pt_index = i * 2
        pointer = backbone.layer[pt_index]
        prefix = f'MobilenetV1/Conv2d_{tf_index}_depthwise/'
        tf_to_pt_map[prefix + 'depthwise_weights'] = pointer.convolution.weight
        tf_to_pt_map[prefix + 'BatchNorm/beta'] = pointer.normalization.bias
        tf_to_pt_map[prefix + 'BatchNorm/gamma'] = pointer.normalization.weight
        tf_to_pt_map[prefix + 'BatchNorm/moving_mean'] = pointer.normalization.running_mean
        tf_to_pt_map[prefix + 'BatchNorm/moving_variance'] = pointer.normalization.running_var
        pointer = backbone.layer[pt_index + 1]
        prefix = f'MobilenetV1/Conv2d_{tf_index}_pointwise/'
        tf_to_pt_map[prefix + 'weights'] = pointer.convolution.weight
        tf_to_pt_map[prefix + 'BatchNorm/beta'] = pointer.normalization.bias
        tf_to_pt_map[prefix + 'BatchNorm/gamma'] = pointer.normalization.weight
        tf_to_pt_map[prefix + 'BatchNorm/moving_mean'] = pointer.normalization.running_mean
        tf_to_pt_map[prefix + 'BatchNorm/moving_variance'] = pointer.normalization.running_var
    if isinstance(model, MobileNetV1ForImageClassification):
        prefix = 'MobilenetV1/Logits/Conv2d_1c_1x1/'
        tf_to_pt_map[prefix + 'weights'] = model.classifier.weight
        tf_to_pt_map[prefix + 'biases'] = model.classifier.bias
    return tf_to_pt_map