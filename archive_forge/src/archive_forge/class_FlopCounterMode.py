import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
class FlopCounterMode(TorchDispatchMode):
    """
    ``FlopCounterMode`` is a context manager that counts the number of flops within its context.

    It does this using a ``TorchDispatchMode``.

    It also supports hierarchical output by passing a module (or list of
    modules) to FlopCounterMode on construction. If you do not need hierarchical
    output, you do not need to use it with a module.

    Example usage

    .. code-block:: python

        mod = ...
        flop_counter = FlopCounterMode(mod)
        with flop_counter:
            mod.sum().backward()

    """

    def __init__(self, mods: Optional[Union[torch.nn.Module, List[torch.nn.Module]]]=None, depth: int=2, display: bool=True, custom_mapping: Optional[Dict[Any, Any]]=None):
        self.flop_counts: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.depth = depth
        self.parents = ['Global']
        self.display = display
        if custom_mapping is None:
            custom_mapping = {}
        if isinstance(mods, torch.nn.Module):
            mods = [mods]
        self.mods = mods
        self._module_to_forward_hook_handles: Dict[nn.Module, _ForwardHookHandles] = {}
        self.flop_registry = {**flop_registry, **{k: v if getattr(v, '_get_raw', False) else shape_wrapper(v) for k, v in custom_mapping.items()}}

    def _register_forward_hooks(self):
        if self.mods is None:
            return
        for mod in self.mods:
            prefix = type(mod).__name__
            for name, module in dict(mod.named_modules()).items():
                if name == '':
                    name = prefix
                else:
                    name = '.'.join([prefix, name])
                forward_pre_hook_handle = module.register_forward_pre_hook(self._enter_module(name))
                forward_hook_handle = module.register_forward_hook(self._exit_module(name))
                self._module_to_forward_hook_handles[module] = _ForwardHookHandles(forward_pre_hook_handle, forward_hook_handle)

    def _deregister_forward_hooks(self):
        for forward_hook_handles in self._module_to_forward_hook_handles.values():
            forward_hook_handles[0].remove()
            forward_hook_handles[1].remove()
        self._module_to_forward_hook_handles.clear()

    def _enter_module(self, name):

        def f(module, inputs):
            out = _pytreeify_preserve_structure(self._create_pre_module(name))(inputs)
            return out
        return f

    def _exit_module(self, name):

        def f(module, inputs, outputs):
            outputs = _pytreeify_preserve_structure(self._create_post_module(name))(outputs)
            return outputs
        return f

    def _create_post_module(self, name):

        class PushState(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *args):
                assert self.parents[-1] == name
                self.parents.pop()
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.parents.append(name)
                return grad_outs
        return PushState.apply

    def _create_pre_module(self, name):

        class PopState(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *args):
                self.parents.append(name)
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                assert self.parents[-1] == name
                self.parents.pop()
                return grad_outs
        return PopState.apply

    def get_total_flops(self) -> int:
        return sum(self.flop_counts['Global'].values())

    def get_flop_counts(self) -> Dict[str, Dict[Any, int]]:
        """Return the flop counts as a dictionary of dictionaries.

        The outer
        dictionary is keyed by module name, and the inner dictionary is keyed by
        operation name.

        Returns:
            Dict[str, Dict[Any, int]]: The flop counts as a dictionary.
        """
        return dict(self.flop_counts)

    def get_table(self, depth=None):
        if depth is None:
            depth = self.depth
        if depth is None:
            depth = 999999
        import tabulate
        tabulate.PRESERVE_WHITESPACE = True
        header = ['Module', 'FLOP', '% Total']
        values = []
        global_flops = self.get_total_flops()
        global_suffix = get_suffix_str(global_flops)
        is_global_subsumed = False

        def process_mod(mod_name, depth):
            nonlocal is_global_subsumed
            total_flops = sum(self.flop_counts[mod_name].values())
            is_global_subsumed |= total_flops >= global_flops
            padding = ' ' * depth
            values = []
            values.append([padding + mod_name, convert_num_with_suffix(total_flops, global_suffix), convert_to_percent_str(total_flops, global_flops)])
            for k, v in self.flop_counts[mod_name].items():
                values.append([padding + ' - ' + str(k), convert_num_with_suffix(v, global_suffix), convert_to_percent_str(v, global_flops)])
            return values
        for mod in self.flop_counts.keys():
            if mod == 'Global':
                continue
            mod_depth = mod.count('.') + 1
            if mod_depth > depth:
                continue
            cur_values = process_mod(mod, mod_depth - 1)
            for value in cur_values:
                values.append(value)
        if 'Global' in self.flop_counts and (not is_global_subsumed):
            for idx, value in enumerate(values):
                values[idx][0] = ' ' + values[idx][0]
            values = process_mod('Global', 0) + values
        if len(values) == 0:
            values = [['Global', '0', '0%']]
        return tabulate.tabulate(values, headers=header, colalign=('left', 'right', 'right'))

    def __enter__(self):
        self.flop_counts.clear()
        self._register_forward_hooks()
        super().__enter__()
        return self

    def __exit__(self, *args):
        if self.display:
            print(self.get_table(self.depth))
        self._deregister_forward_hooks()
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out=out)
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count
        return out