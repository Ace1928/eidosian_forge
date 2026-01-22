import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
def _get_node_label(self, module: torch.fx.GraphModule, node: torch.fx.Node, skip_node_names_in_args: bool, parse_stack_trace: bool) -> str:

    def _get_str_for_args_kwargs(arg):
        if isinstance(arg, tuple):
            prefix, suffix = ('|args=(\\l', ',\\n)\\l')
            arg_strs_list = [_format_arg(a, max_list_len=8) for a in arg]
        elif isinstance(arg, dict):
            prefix, suffix = ('|kwargs={\\l', ',\\n}\\l')
            arg_strs_list = [f'{k}: {_format_arg(v, max_list_len=8)}' for k, v in arg.items()]
        else:
            return ''
        if skip_node_names_in_args:
            arg_strs_list = [a for a in arg_strs_list if '%' not in a]
        if len(arg_strs_list) == 0:
            return ''
        arg_strs = prefix + ',\\n'.join(arg_strs_list) + suffix
        if len(arg_strs_list) == 1:
            arg_strs = arg_strs.replace('\\l', '').replace('\\n', '')
        return arg_strs.replace('{', '\\{').replace('}', '\\}')
    label = '{' + f'name=%{node.name}|op_code={node.op}\n'
    if node.op == 'call_module':
        leaf_module = self._get_leaf_node(module, node)
        label += '\\n' + self._typename(leaf_module) + '\\n|'
        extra = ''
        if hasattr(leaf_module, '__constants__'):
            extra = '\\n'.join([f'{c}: {getattr(leaf_module, c)}' for c in leaf_module.__constants__])
        label += extra + '\\n'
    else:
        label += f'|target={self._typename(node.target)}' + '\\n'
        if len(node.args) > 0:
            label += _get_str_for_args_kwargs(node.args)
        if len(node.kwargs) > 0:
            label += _get_str_for_args_kwargs(node.kwargs)
        label += f'|num_users={len(node.users)}' + '\\n'
    tensor_meta = node.meta.get('tensor_meta')
    label += self._tensor_meta_to_label(tensor_meta)
    buf_meta = node.meta.get('buf_meta', None)
    if buf_meta is not None:
        label += f'|buf={buf_meta.name}' + '\\n'
        label += f'|n_origin={buf_meta.n_origin}' + '\\n'
    if parse_stack_trace and node.stack_trace is not None:
        parsed_stack_trace = _parse_stack_trace(node.stack_trace)
        fname = self._shorten_file_name(parsed_stack_trace.file)
        label += f'|file={fname}:{parsed_stack_trace.lineno} {parsed_stack_trace.code}' + '\\n'
    return label + '}'