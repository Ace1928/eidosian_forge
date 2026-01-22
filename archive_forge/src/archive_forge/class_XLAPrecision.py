import os
from typing import Any, Literal
import torch
from typing_extensions import get_args, override
from lightning_fabric.accelerators.xla import _XLA_AVAILABLE
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.utilities.types import Optimizable
class XLAPrecision(Precision):
    """Plugin for training with XLA.

    Args:
        precision: Full precision (32-true) or half precision (16-true, bf16-true).

    Raises:
        ValueError:
            If unsupported ``precision`` is provided.

    """

    def __init__(self, precision: _PRECISION_INPUT) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(f'`precision={precision!r})` is not supported in XLA. `precision` must be one of: {supported_precision}.')
        self.precision = precision
        if precision == '16-true':
            os.environ['XLA_USE_F16'] = '1'
            self._desired_dtype = torch.float16
        elif precision == 'bf16-true':
            os.environ['XLA_USE_BF16'] = '1'
            self._desired_dtype = torch.bfloat16
        else:
            self._desired_dtype = torch.float32

    @override
    def optimizer_step(self, optimizer: Optimizable, **kwargs: Any) -> Any:
        import torch_xla.core.xla_model as xm
        return xm.optimizer_step(optimizer, optimizer_args=kwargs, barrier=True)

    @override
    def teardown(self) -> None:
        os.environ.pop('XLA_USE_BF16', None)
        os.environ.pop('XLA_USE_F16', None)