import copy
import dataclasses
import functools
from typing import (
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from .graph_signature import (  # noqa: F401
def _get_updated_graph_signature(old_signature: ExportGraphSignature, new_gm: torch.fx.GraphModule) -> ExportGraphSignature:
    """
            Update the graph signature's user_input/user_outputs.
            """
    new_input_specs = []
    for i, node in enumerate(new_gm.graph.nodes):
        if node.op != 'placeholder':
            break
        assert i < len(old_signature.input_specs), 'Number of inputs changed after transformation'
        old_input_spec = old_signature.input_specs[i]
        arg = old_input_spec.arg if isinstance(old_input_spec.arg, ConstantArgument) else type(old_input_spec.arg)(node.name)
        new_input_specs.append(InputSpec(old_input_spec.kind, arg, old_input_spec.target))
    output_node = list(new_gm.graph.nodes)[-1]
    assert output_node.op == 'output'
    new_output_specs = []
    for i, node in enumerate(output_node.args[0]):
        assert i < len(old_signature.output_specs), 'Number of outputs changed after transformation'
        old_output_spec = old_signature.output_specs[i]
        arg = old_output_spec.arg if isinstance(old_output_spec.arg, ConstantArgument) else type(old_output_spec.arg)(node.name)
        new_output_specs.append(OutputSpec(old_output_spec.kind, arg, old_output_spec.target))
    new_signature = ExportGraphSignature(input_specs=new_input_specs, output_specs=new_output_specs)
    return new_signature