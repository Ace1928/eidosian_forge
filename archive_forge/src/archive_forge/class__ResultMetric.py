from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, cast
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torchmetrics import Metric
from typing_extensions import TypedDict, override
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning_fabric.utilities.distributed import _distributed_is_initialized
from lightning_fabric.utilities.imports import _TORCH_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities.data import extract_batch_size
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_1_0_0
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
from pytorch_lightning.utilities.warnings import PossibleUserWarning
class _ResultMetric(Metric):
    """Wraps the value provided to `:meth:`~pytorch_lightning.core.LightningModule.log`"""

    def __init__(self, metadata: _Metadata, is_tensor: bool) -> None:
        super().__init__()
        self.is_tensor = is_tensor
        self.meta = metadata
        self.has_reset = False
        if is_tensor:
            if metadata.is_max_reduction:
                default = float('-inf')
            elif metadata.is_min_reduction:
                default = float('inf')
            else:
                default = 0.0
            self.add_state('value', torch.tensor(default, dtype=_get_default_dtype()), dist_reduce_fx=torch.sum)
            if self.meta.is_mean_reduction:
                self.cumulated_batch_size: Tensor
                self.add_state('cumulated_batch_size', torch.tensor(0), dist_reduce_fx=torch.sum)
        self._forward_cache: Optional[Any] = None

    @override
    def update(self, value: _VALUE, batch_size: int) -> None:
        if self.is_tensor:
            value = cast(Tensor, value)
            dtype = _get_default_dtype()
            if not torch.is_floating_point(value):
                warning_cache.warn(f"You called `self.log({self.meta.name!r}, ...)` in your `{self.meta.fx}` but the value needs to be floating to be reduced. Converting it to {dtype}. You can silence this warning by converting the value to floating point yourself. If you don't intend to reduce the value (for instance when logging the global step or epoch) then you can use `self.logger.log_metrics({{{self.meta.name!r}: ...}})` instead.")
                value = value.to(dtype)
            if value.dtype not in (torch.float32, torch.float64):
                value = value.to(dtype)
            if self.meta.on_step:
                self._forward_cache = self.meta.sync(value.clone())
                if not self.meta.on_epoch:
                    self.value = self._forward_cache
                    return
            if self.meta.is_mean_reduction:
                self.value = self.value + value * batch_size
                self.cumulated_batch_size = self.cumulated_batch_size + batch_size
            elif self.meta.is_max_reduction or self.meta.is_min_reduction:
                self.value = self.meta.reduce_fx(self.value, value)
            elif self.meta.is_sum_reduction:
                self.value = self.value + value
        else:
            value = cast(Metric, value)
            self.value = value
            self._forward_cache = value._forward_cache

    @override
    def compute(self) -> Tensor:
        if self.is_tensor:
            value = self.meta.sync(self.value.clone())
            if self.meta.is_mean_reduction:
                cumulated_batch_size = self.meta.sync(self.cumulated_batch_size)
                return value / cumulated_batch_size
            return value
        return self.value.compute()

    @override
    def reset(self) -> None:
        if self.is_tensor:
            super().reset()
        else:
            self.value.reset()
        self.has_reset = True

    @override
    def forward(self, value: _VALUE, batch_size: int) -> None:
        if self.meta.enable_graph:
            with torch.no_grad():
                self.update(value, batch_size)
        else:
            self.update(value, batch_size)

    @override
    def _wrap_compute(self, compute: Any) -> Any:

        @wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Optional[Any]:
            update_called = self.update_called if _TORCHMETRICS_GREATER_EQUAL_1_0_0 else self._update_called
            if not update_called:
                rank_zero_warn(f'The ``compute`` method of metric {self.__class__.__name__} was called before the ``update`` method which may lead to errors, as metric states have not yet been updated.')
            if self._computed is not None:
                return self._computed
            self._computed = compute(*args, **kwargs)
            return self._computed
        return wrapped_func

    @override
    def __setattr__(self, key: str, value: Any) -> None:
        object.__setattr__(self, key, value)

    @override
    def __repr__(self) -> str:
        state = f'{repr(self.meta.name)}, value={self.value}'
        if self.is_tensor and self.meta.is_mean_reduction:
            state += f', cumulated_batch_size={self.cumulated_batch_size}'
        return f'{self.__class__.__name__}({state})'

    @override
    def to(self, *args: Any, **kwargs: Any) -> '_ResultMetric':
        d = self.__dict__
        if _TORCH_GREATER_EQUAL_2_0:
            d = dict(d)
        self.__dict__.update(apply_to_collection(d, (Tensor, Metric), move_data_to_device, *args, **kwargs))
        return self