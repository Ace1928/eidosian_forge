from typing import Any, ContextManager, Literal
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
class HalfPrecision(Precision):
    """Plugin for training with half precision.

    Args:
        precision: Whether to use ``torch.float16`` (``'16-true'``) or ``torch.bfloat16`` (``'bf16-true'``).

    """
    precision: Literal['bf16-true', '16-true'] = '16-true'

    def __init__(self, precision: Literal['bf16-true', '16-true']='16-true') -> None:
        self.precision = precision
        self._desired_input_dtype = torch.bfloat16 if precision == 'bf16-true' else torch.float16

    @override
    def convert_module(self, module: Module) -> Module:
        return module.to(dtype=self._desired_input_dtype)

    @override
    def tensor_init_context(self) -> ContextManager:
        return _DtypeContextManager(self._desired_input_dtype)

    @override
    def module_init_context(self) -> ContextManager:
        return self.tensor_init_context()

    @override
    def forward_context(self) -> ContextManager:
        return self.tensor_init_context()

    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype)

    @override
    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())