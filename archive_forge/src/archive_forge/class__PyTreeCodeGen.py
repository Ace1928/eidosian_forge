import collections
from collections import defaultdict
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
import torch.utils._pytree as pytree
from . import _pytree as fx_pytree
from ._compatibility import compatibility
import contextlib
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type
from dataclasses import dataclass
from contextlib import contextmanager
import copy
import enum
import torch
import keyword
import re
import builtins
import math
import warnings
import inspect
class _PyTreeCodeGen(CodeGen):

    def __init__(self, pytree_info: _PyTreeInfo):
        super().__init__()
        self.pytree_info: _PyTreeInfo = pytree_info

    def process_inputs(self, *inputs: Any) -> Any:
        flat_args = pytree.arg_tree_leaves(*inputs)
        return flat_args

    def process_outputs(self, out: Any) -> Any:
        if self.pytree_info is None or self.pytree_info.out_spec is None:
            return out
        if not isinstance(out, (list, tuple)):
            out = [out]
        assert self.pytree_info.out_spec is not None
        return pytree.tree_unflatten(out, self.pytree_info.out_spec)

    def gen_fn_def(self, free_vars, maybe_return_annotation):
        if self.pytree_info is None:
            return super().gen_fn_def(free_vars, maybe_return_annotation)
        fn_args = self.pytree_info.orig_args
        has_orig_self = fn_args[0] == 'self' if len(fn_args) > 0 else False
        if has_orig_self:
            free_vars.insert(0, 'self')
        fn_definition = super().gen_fn_def(fn_args[:], maybe_return_annotation)
        if len(free_vars) > 0:
            has_args_kwargs_tuple = self.pytree_info.in_spec.type == tuple and len(self.pytree_info.in_spec.children_specs) == 2 and (self.pytree_info.in_spec.children_specs[0].type == tuple) and (self.pytree_info.in_spec.children_specs[1].type == dict)
            fn_kwargs = '{}'
            fn_signature = f'[{', '.join(fn_args)}], self._in_spec'
            if has_args_kwargs_tuple:
                count_args = len(self.pytree_info.in_spec.children_specs[0].children_specs)
                fn_args = self.pytree_info.orig_args[:count_args]
                fn_kwargs = '{' + ', '.join((f"'{k}':{v}" for k, v in zip(self.pytree_info.in_spec.children_specs[1].context, self.pytree_info.orig_args[count_args:]))) + '}'
                fn_signature = f'([{', '.join(fn_args)}], {fn_kwargs}), self._in_spec'
            without_annotation = [x.split(':')[0] for x in free_vars]
            has_annotation = [x + '; ' for x in free_vars if ':' in x]
            if len(has_annotation) > 0:
                fn_definition += '\n    ' + ''.join(has_annotation) + '\n'
            fn_definition += f'\n    {', '.join(without_annotation)}, = fx_pytree.tree_flatten_spec({fn_signature})'
        return fn_definition

    def generate_output(self, output_args):
        if self.pytree_info and self.pytree_info.out_spec:
            return f'return pytree.tree_unflatten({repr(output_args)}, self._out_spec)'
        else:
            return super().generate_output(output_args)