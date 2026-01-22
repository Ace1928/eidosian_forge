from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, TypeVar, Union
import torch
from typing_extensions import override
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
def _plugin_to_compute_dtype(plugin: 'Precision') -> torch.dtype:
    from lightning_fabric.plugins import BitsandbytesPrecision, DeepSpeedPrecision, DoublePrecision, FSDPPrecision, HalfPrecision, MixedPrecision, Precision, TransformerEnginePrecision, XLAPrecision
    if not isinstance(plugin, Precision):
        raise RuntimeError(f'Expected a precision plugin, got {plugin}')
    if isinstance(plugin, BitsandbytesPrecision):
        return plugin.dtype
    if isinstance(plugin, (HalfPrecision, MixedPrecision)):
        return plugin._desired_input_dtype
    if isinstance(plugin, DoublePrecision):
        return torch.double
    if isinstance(plugin, (XLAPrecision, DeepSpeedPrecision)):
        return plugin._desired_dtype
    if isinstance(plugin, TransformerEnginePrecision):
        return torch.int8
    if isinstance(plugin, FSDPPrecision):
        return plugin.mixed_precision_config.reduce_dtype or torch.float32
    if isinstance(plugin, Precision):
        return torch.float32
    raise NotImplementedError(plugin)