from __future__ import annotations
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union, List
import torch
from .fake_quantize import (
from .observer import (
from .qconfig import (
def _get_default_qconfig_mapping(is_qat: bool, backend: str, version: int) -> QConfigMapping:
    """
    Return the default QConfigMapping for the given quantization type and backend.
    """
    if is_qat:
        qconfig = get_default_qat_qconfig(backend, version)
    else:
        qconfig = get_default_qconfig(backend, version)
    default_weight = default_weight_fake_quant if is_qat else default_weight_observer
    if backend in ('fbgemm', 'x86'):
        qconfig_transpose = QConfig(activation=qconfig.activation, weight=default_weight)
    else:
        qconfig_transpose = qconfig
    qconfig_layernorm = QConfig(activation=qconfig.activation, weight=default_placeholder_observer)
    qconfig_mapping = QConfigMapping().set_global(qconfig).set_object_type('reshape', default_reuse_input_qconfig).set_object_type(torch.nn.ConvTranspose1d, qconfig_transpose).set_object_type(torch.nn.ConvTranspose2d, qconfig_transpose).set_object_type(torch.nn.ConvTranspose3d, qconfig_transpose).set_object_type(torch.nn.functional.conv_transpose1d, qconfig_transpose).set_object_type(torch.nn.functional.conv_transpose2d, qconfig_transpose).set_object_type(torch.nn.functional.conv_transpose3d, qconfig_transpose).set_object_type(torch.nn.functional.layer_norm, qconfig_layernorm).set_object_type(torch.nn.LayerNorm, qconfig_layernorm).set_object_type(torch.nn.PReLU, default_quint8_weight_qconfig)
    fixed_qparams_observer_to_qconfig: Dict[Any, QConfigAny] = {}
    for fixed_qparams_op, observer in _FIXED_QPARAMS_OP_TO_OBSERVER.items():
        if observer in fixed_qparams_observer_to_qconfig:
            fixed_qparams_qconfig = fixed_qparams_observer_to_qconfig[observer]
        else:
            if is_qat:
                activation = FixedQParamsFakeQuantize.with_args(observer=observer)
            else:
                activation = observer
            fixed_qparams_qconfig = QConfig(activation=activation, weight=default_weight)
            fixed_qparams_observer_to_qconfig[observer] = fixed_qparams_qconfig
        qconfig_mapping.set_object_type(fixed_qparams_op, fixed_qparams_qconfig)
    return qconfig_mapping