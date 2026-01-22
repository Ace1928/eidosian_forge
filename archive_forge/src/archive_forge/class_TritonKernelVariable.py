import functools
import inspect
import itertools
import types
from typing import Dict, List
import torch
from .. import variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented, Unsupported
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import make_cell
from .base import typestr, VariableTracker
class TritonKernelVariable(VariableTracker):

    def __init__(self, kernel, kernel_idx, grid, **kwargs):
        from triton.runtime.autotuner import Autotuner
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        super().__init__(**kwargs)
        assert kernel is not None
        self.kernel = kernel
        self.kernel_idx = kernel_side_table.add_kernel(kernel)
        assert kernel_idx is None or self.kernel_idx == kernel_idx
        self.grid = grid
        if isinstance(kernel, Autotuner):
            defaults = inspect.signature(Autotuner).parameters
            if 'warmup' in defaults and defaults['warmup'].default != kernel.warmup or ('rep' in defaults and defaults['rep'].default != kernel.rep) or ('prune_configs_by' in defaults and defaults['prune_configs_by'].default != kernel.early_config_prune):
                raise Unsupported('Only configs and keys are supported for triton.autotune')

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from triton.runtime.autotuner import Autotuner
        from .constant import ConstantVariable
        from .dicts import ConstDictVariable
        from .lists import BaseListVariable
        if self.grid is None:
            raise Unsupported('Triton kernels should always be called with a grid')
        normalized_args = {**dict(zip(self.kernel.arg_names, args)), **kwargs}
        configs = [config.kwargs for config in self.kernel.configs] if isinstance(self.kernel, Autotuner) else [{}]
        grids = []
        for config_args in configs:
            grid = self.grid
            if isinstance(grid, (NestedUserFunctionVariable, UserFunctionVariable)):
                config_args = {k: ConstantVariable.create(v) for k, v in config_args.items()}
                meta = ConstDictVariable({**normalized_args, **config_args}, dict)
                grid = grid.call_function(tx, [meta], {})
            if isinstance(grid, BaseListVariable):
                grids.append(grid.as_proxy())
            else:
                unimplemented(f'grid for the triton kernel is {type(grid)}')
        for i in range(len(grids)):
            if not isinstance(grids[i], tuple):
                raise Unsupported('Only tuple grids are supported')
            if len(grids[i]) == 1:
                grids[i] = (grids[i][0], 1, 1)
            elif len(grids[i]) == 2:
                grids[i] = (grids[i][0], grids[i][1], 1)
            elif len(grids[i]) > 3:
                raise Unsupported('Grid can have at most rank 3')
        assert len(grids) != 0
        if len(set(grids)) == 1:
            grids = [grids[0]]
        from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
        meta = ConstDictVariable(normalized_args, dict)
        tx.output.create_proxy('call_function', triton_kernel_wrapper_mutation, (), {'kernel_idx': self.kernel_idx, 'grid': grids, 'kwargs': meta.as_proxy()})
        return variables.ConstantVariable(None)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if name == '__getitem__':
            if self.grid is not None or len(args) != 1:
                raise Unsupported('Triton kernels should be called with only a single grid')
            return TritonKernelVariable(kernel=self.kernel, kernel_idx=self.kernel_idx, grid=args[0])
        elif name == 'run':
            if 'grid' not in kwargs:
                raise Unsupported('Triton kernel requires to be called with a grid')
            grid = kwargs.pop('grid')
            return self.clone(grid=grid).call_function(tx, args, kwargs)
        return super().call_method(tx, name, args, kwargs)