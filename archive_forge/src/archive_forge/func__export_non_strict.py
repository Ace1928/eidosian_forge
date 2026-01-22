import copy
import dataclasses
import functools
import io
import json
import pathlib
import re
import sys
import types
import warnings
import weakref
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch
import sympy
import torch
import torch._dynamo
import torch.fx
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.exc import UserError, UserErrorType
from torch._dynamo.source import ConstantSource
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._functorch.aot_autograd import aot_export_module, GraphSignature
from torch._functorch.eager_transforms import functionalize
from torch._guards import detect_fake_mode
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import _create_constraint, _Dim, Constraint
from torch.export.exported_program import (
from torch.export.graph_signature import (
from torch.fx import traceback as fx_traceback
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRangeError, ValueRanges
from .exported_program import (
from .passes.add_runtime_assertions_for_constraints_pass import (
from .passes.lift_constant_tensor_pass import lift_constant_tensor_pass
from .passes.remove_runtime_assertions import _RemoveRuntimeAssertionsPass
from .passes.replace_sym_size_ops_pass import _replace_sym_size_ops_pass
from .passes.replace_view_ops_with_view_copy_ops_pass import (
from .wrappers import _wrap_submodules
def _export_non_strict(mod, fake_args, fake_kwargs, fake_params_buffers, *, transform=lambda x: x):
    with torch.nn.utils.stateless._reparametrize_module(mod, fake_params_buffers):
        gm, graph_signature = transform(aot_export_module)(mod, (*fake_args, *fake_kwargs.values()), trace_joint=False)
    flat_args = pytree.tree_leaves((fake_args, fake_kwargs))
    index = 0
    total_param_buffers = len(graph_signature.parameters) + len(graph_signature.buffers)
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            if index >= total_param_buffers:
                user_arg = flat_args[index - total_param_buffers]
                if not isinstance(user_arg, torch.Tensor):
                    node.meta['val'] = user_arg
            index += 1
    is_joint = graph_signature.backward_signature is not None

    def make_argument_spec(node) -> ArgumentSpec:
        assert 'val' in node.meta, f"{node} has no 'val' metadata field"
        val = node.meta['val']
        if isinstance(val, FakeTensor):
            return TensorArgument(name=node.name)
        elif isinstance(val, torch.SymInt):
            return SymIntArgument(name=node.name)
        else:
            return ConstantArgument(value=val)
    input_specs, output_specs = _sig_to_specs(user_inputs=set(graph_signature.user_inputs), inputs_to_parameters=graph_signature.inputs_to_parameters, inputs_to_buffers=graph_signature.inputs_to_buffers, user_outputs=set(graph_signature.user_outputs), buffer_mutations=graph_signature.buffers_to_mutate, user_input_mutations=gm.meta.get('user_inputs_to_mutate', {}), grad_params=graph_signature.backward_signature.gradients_to_parameters if is_joint else {}, grad_user_inputs=graph_signature.backward_signature.gradients_to_user_inputs if is_joint else {}, loss_output=graph_signature.backward_signature.loss_output if is_joint else None, inputs=[make_argument_spec(node) for node in gm.graph.nodes if node.op == 'placeholder'], outputs=[make_argument_spec(node) for node in pytree.tree_leaves(next(iter(reversed(gm.graph.nodes))).args)])
    export_graph_signature = ExportGraphSignature(input_specs=input_specs, output_specs=output_specs)
    tensor_constants = lift_constant_tensor_pass(gm, export_graph_signature)

    @dataclasses.dataclass
    class _ExportedProgramNonStrict:
        gm: torch.fx.GraphModule
        sig: ExportGraphSignature
        tensor_constants: Dict[str, torch.Tensor]
    return _ExportedProgramNonStrict(gm, export_graph_signature, tensor_constants)