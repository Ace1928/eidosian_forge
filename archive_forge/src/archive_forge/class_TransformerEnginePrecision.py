import logging
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, ContextManager, Literal, Mapping, Optional, Union
import torch
from lightning_utilities import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import (
from lightning_fabric.utilities.rank_zero import rank_zero_info, rank_zero_warn
class TransformerEnginePrecision(Precision):
    """Plugin for training with fp8 precision via nvidia's
    `Transformer Engine <https://docs.nvidia.com/deeplearning/transformer-engine>`__.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        weights_dtype: The weights dtype to use.
        recipe: Recipe for the DelayedScaling
            `configuration <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.DelayedScaling>`__.
            In dict format or the dataclass format.
        replace_layers: Whether to replace ``Linear`` and ``LayerNorm`` layers automatically with their Transformer
            Engine alternatives. Note that they don't subclass the torch equivalents so checks like
            ``isinstance(l, torch.nn.Linear)`` will not pass.
        fallback_compute_dtype: The compute dtype to use for operations that don't support fp8 autocast. Defaults to the
            same as ``weights_dtype``.

    .. note::

        Support for FP8 in the linear layers with this plugin is currently limited to tensors
        with shapes where the dimensions are divisible by 8 and 16 respectively. You might want to add padding to your
        inputs to conform to this restriction.

    """
    precision: Literal['transformer-engine', 'transformer-engine-float16'] = 'transformer-engine'

    def __init__(self, *, weights_dtype: torch.dtype, recipe: Optional[Union[Mapping[str, Any], 'DelayedScaling']]=None, replace_layers: Optional[bool]=None, fallback_compute_dtype: Optional[torch.dtype]=None) -> None:
        if not _TRANSFORMER_ENGINE_AVAILABLE:
            raise ModuleNotFoundError(str(_TRANSFORMER_ENGINE_AVAILABLE))
        from transformer_engine.common.recipe import DelayedScaling
        if recipe is None:
            recipe = DelayedScaling()
        elif isinstance(recipe, Mapping):
            recipe = dict(recipe)
            if 'fp8_format' in recipe:
                from transformer_engine.common.recipe import Format
                recipe['fp8_format'] = getattr(Format, recipe['fp8_format'])
            recipe = DelayedScaling(**recipe)
        self.weights_dtype = weights_dtype
        self.recipe = recipe
        self.replace_layers = replace_layers
        self.fallback_compute_dtype = fallback_compute_dtype or weights_dtype

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        if any(('transformer_engine.pytorch' in m.__module__ for m in module.modules())):
            if self.replace_layers is True:
                rank_zero_info('`TransformerEnginePrecision(replace_layers=True)` is set but the model already contains TransformerEngine layers. Skipping')
        elif self.replace_layers in (None, True):
            _convert_layers(module)
        module = module.to(dtype=self.weights_dtype)
        return module

    @override
    def tensor_init_context(self) -> ContextManager:
        return _DtypeContextManager(self.weights_dtype)

    @override
    def module_init_context(self) -> ContextManager:
        dtype_ctx = self.tensor_init_context()
        stack = ExitStack()
        if self.replace_layers:
            import transformer_engine.pytorch as te
            context_manager = _ClassReplacementContextManager({'torch.nn.Linear': te.Linear, 'torch.nn.LayerNorm': te.LayerNorm})
            stack.enter_context(context_manager)
        stack.enter_context(dtype_ctx)
        return stack

    @override
    def forward_context(self) -> ContextManager:
        dtype_ctx = _DtypeContextManager(self.weights_dtype)
        fallback_autocast_ctx = torch.autocast(device_type='cuda', dtype=self.fallback_compute_dtype)
        import transformer_engine.pytorch as te
        autocast_ctx = te.fp8_autocast(enabled=True, fp8_recipe=self.recipe)
        stack = ExitStack()
        stack.enter_context(dtype_ctx)
        stack.enter_context(fallback_autocast_ctx)
        stack.enter_context(autocast_ctx)
        return stack

    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self.weights_dtype)

    @override
    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())