import copy
import operator
from copy import deepcopy
from typing import cast, Dict, List, Optional, Union
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.exported_program import (
from torch.fx import GraphModule
from .utils import _check_input_constraints_pre_hook
class _UnflattenedModule(torch.fx.GraphModule):

    def __init__(self, export_module: ExportedProgram):
        if export_module.graph_signature.backward_signature is not None:
            raise ValueError('Unflattening on JointExportModule NYI')
        super().__init__({}, torch.fx.Graph(), '_UnflattenedModule')
        export_graph = deepcopy(export_module.graph)
        self.graph_signature = deepcopy(export_module.graph_signature)
        self.module_call_graph = deepcopy(export_module.module_call_graph)
        _inplace_buffer_mutations(export_graph, self.graph_signature)
        _outline_submodules(export_graph, self)
        self.range_constraints = export_module.range_constraints
        self.equality_constraints = export_module.equality_constraints
        state_dict = export_module.state_dict
        for name in self.graph_signature.parameters:
            cloned = state_dict[name].clone()
            _assign_attr(cloned, self, name, is_parameter=True)
        for name in self.graph_signature.buffers:
            cloned = state_dict[name].clone()
            _assign_attr(cloned, self, name, is_parameter=False)
        inputs_to_state: Dict[str, str] = {**self.graph_signature.inputs_to_parameters, **self.graph_signature.inputs_to_buffers}
        _sink_params(self, inputs_to_state, [])
        for module in self.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != 'placeholder':
                    continue
                assert node.name not in inputs_to_state

    def __call__(self, *args, **kwargs):
        flat_args, in_spec = pytree.tree_flatten((args, kwargs))
        assert self.module_call_graph[0].fqn == ''
        signature = self.module_call_graph[0].signature
        if in_spec != signature.in_spec:
            raise TypeError(f"Input treespec does not match with exported module's. Are you sure you are calling this with the right arguments? Input treespec: {in_spec}. ", f'Exported module treespec: {signature.in_spec}')
        tree_out = super().__call__(*flat_args)
        return pytree.tree_unflatten(tree_out, signature.out_spec)