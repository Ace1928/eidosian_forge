import contextlib
import functools
import itertools
import logging
from typing import Dict, List, Optional
import torch._C
import torch.fx
import torch.nn
import torch.onnx.operators
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import deepcopy_to_fake_tensor, get_fake_value, get_real_value
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builtin import BuiltinVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.tensor import SymNodeVariable
from torch._guards import Source
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree
from ..exc import (
from ..source import FSDPNNModuleSource, GetItemSource, NNModuleSource
from ..utils import proxy_args_kwargs
from .dicts import ConstDictVariable
from .lists import ListVariable, TupleVariable
from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable
class CondHigherOrderVariable(TorchHigherOrderOperatorVariable):

    @raise_hard_error_if_graph_break(reason="Cond doesn't work unless it is captured completely with torch.compile.")
    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from . import ConstantVariable, ListVariable, NestedUserFunctionVariable, TensorVariable, UserFunctionVariable
        from .builder import wrap_fx_proxy
        args, kwargs = VariableTracker.apply(lambda x: x.realize(), (args, kwargs))
        for i, k in enumerate(['pred', 'true_fn', 'false_fn', 'operands']):
            if (v := kwargs.pop(k, None)):
                assert i == len(args), 'did not provide the right number of non-keyword args'
                args.append(v)
        if kwargs:
            unimplemented(f'torch.cond: Got unexpected kwargs: {list(kwargs.keys())}')
        if len(args) != 4:
            unimplemented(f'Expected 4 arguments but got {len(args)}.\nUsage: cond(pred, true_fn, false_fn, operands)')
        if type(args[0]) not in (ConstantVariable, TensorVariable, SymNodeVariable):
            unimplemented(f'Expected pred to be bool or a boolean tensor with single item but got {str(type(args[0]))} with original python type {str(args[0].python_type())}.')
        if not isinstance(args[3], (ListVariable, TupleVariable)):
            unimplemented(f'Expected a tuple but got {args[3].python_type()}')
        operands = args[3].unpack_var_sequence(tx)
        if not only_consist_of(args[3], (TensorVariable,)):
            unimplemented('Expect operands to be a tuple of pytrees that only consists of tensor leaves.')
        assert isinstance(args[1], (UserFunctionVariable, NestedUserFunctionVariable, NNModuleVariable, UnspecializedNNModuleVariable)), str(type(args[1]))
        assert isinstance(args[2], (UserFunctionVariable, NestedUserFunctionVariable, NNModuleVariable, UnspecializedNNModuleVariable)), str(type(args[2]))
        graph_checkpoint, checkpoint = (tx.output.graph, tx.copy_graphstate())

        def speculate_branch(branch):
            ix = 1 if branch else 2
            (ret_val, ret_treespec), ret_graph, ret_lifted_freevars = speculate_subgraph(tx, args[ix], operands, {}, graph_checkpoint, checkpoint, 'cond', source_target=self.value, manually_set_subgraph_inputs=False, should_flatten_outputs=True)
            if not only_consist_of(ret_val, (TensorVariable,)):
                unimplemented('Expected branches to return a possibly nested list/tuple/dict of tensors but it consists of non tensors.')
            return (ret_val, ret_treespec, ret_graph, ret_lifted_freevars)
        true_r, true_treespec, true_graph, true_lifted_freevars = speculate_branch(True)
        true_nn_modules = tx.copy_graphstate().output.nn_modules
        false_r, false_treespec, false_graph, false_lifted_freevars = speculate_branch(False)
        false_nn_modules = tx.copy_graphstate().output.nn_modules
        same_treespec = _make_inlined(tx, pytree.TreeSpec.__eq__)(true_treespec, false_treespec)
        if not same_treespec.as_python_constant():
            unimplemented('Expected branches to return the same pytree structure.')

        def diff_meta(tensor_vars1, tensor_vars2):
            assert all((isinstance(var, TensorVariable) for var in tensor_vars1 + tensor_vars2))
            all_diffs = []
            for i, (var1, var2) in enumerate(zip(tensor_vars1, tensor_vars2)):
                meta1 = _extract_tensor_metadata(var1.proxy.node.meta['example_value'])
                meta2 = _extract_tensor_metadata(var2.proxy.node.meta['example_value'])
                if meta1 != meta2:
                    all_diffs.append((f'pair{i}:', meta1, meta2))
            return all_diffs
        if (diffs := diff_meta(true_r.unpack_var_sequence(tx), false_r.unpack_var_sequence(tx))):
            unimplemented(f'Expected branches to return tensors with same metadata. [(tensor_pair, difference)...]:{diffs}')

        def dedup_and_sort_lifted_freevars(true_lifted_freevars, false_lifted_freevars):

            def shared_getattrs(true_lifted_proxies, false_lifted_proxies):
                true_targets = {proxy.node.target: proxy for proxy in true_lifted_proxies if proxy.node.op == 'get_attr'}
                true_fn_shared_getattrs = {}
                false_fn_shared_getattrs = {}
                for false_proxy in false_lifted_proxies:
                    if false_proxy.node.op == 'get_attr' and false_proxy.node.target in true_targets:
                        true_proxy = true_targets[false_proxy.node.target]
                        true_fn_shared_getattrs[true_proxy] = true_proxy
                        false_fn_shared_getattrs[false_proxy] = true_proxy
                return (true_fn_shared_getattrs, false_fn_shared_getattrs)
            true_fn_shared_getattrs, false_fn_shared_getattrs = shared_getattrs(true_lifted_freevars.keys(), false_lifted_freevars.keys())
            true_shared_freevars = (true_lifted_freevars.keys() & false_lifted_freevars.keys()).union(true_fn_shared_getattrs.keys())
            false_shared_freevars = (true_lifted_freevars.keys() & false_lifted_freevars.keys()).union(false_fn_shared_getattrs.keys())
            unique_true_freevars = true_lifted_freevars.keys() - true_shared_freevars
            unique_false_freevars = false_lifted_freevars.keys() - false_shared_freevars

            def _sort_by_name(vars):
                return sorted(vars, key=lambda var: var.node.name)
            return (list(_sort_by_name(list(true_shared_freevars))), list(_sort_by_name(list(false_shared_freevars))), list(_sort_by_name(list(unique_true_freevars))), list(_sort_by_name(list(unique_false_freevars))))
        true_shared, false_shared, unique_true, unique_false = dedup_and_sort_lifted_freevars(true_lifted_freevars, false_lifted_freevars)

        def fixup_branch_inps(graph, lifted_freevars, shared, unique_true, unique_false):

            def _insert_or_replace_phs(new_args, name_suffix):
                for arg in new_args:
                    new_ph = graph.placeholder(arg.node.name + name_suffix)
                    if arg in lifted_freevars:
                        old_ph = lifted_freevars[arg].node
                        old_ph.replace_all_uses_with(new_ph)
                        old_ph.users = {}
                        graph.erase_node(old_ph)
            first_not_ph_node = next((node for node in graph.nodes if node.op != 'placeholder'))
            with graph.inserting_before(first_not_ph_node):
                _insert_or_replace_phs(shared, '')
                _insert_or_replace_phs(unique_true, '_true_branch')
                _insert_or_replace_phs(unique_false, '_false_branch')
        fixup_branch_inps(true_graph, true_lifted_freevars, true_shared, unique_true, unique_false)
        fixup_branch_inps(false_graph, false_lifted_freevars, false_shared, unique_true, unique_false)
        true_name = add_subgraph(tx, self.source, 'cond_true', torch.fx.GraphModule(true_nn_modules.nn_modules, true_graph))
        false_name = add_subgraph(tx, self.source, 'cond_false', torch.fx.GraphModule(false_nn_modules.nn_modules, false_graph))
        true_node = make_attr(tx, true_name)
        false_node = make_attr(tx, false_name)
        p_args = (args[0].as_proxy(), true_node, false_node, true_shared + unique_true + unique_false)
        flat_example_value = pytree.tree_map_only(torch.fx.Proxy, lambda a: a.node.meta['example_value'], true_r.as_proxy())
        flat_variable = wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', torch.ops.higher_order.cond, args=tuple(p_args), kwargs={}), example_value=flat_example_value)
        flat_list_variable = BuiltinVariable(list).call_function(tx, [flat_variable], {})
        return _make_inlined(tx, pytree.tree_unflatten)(flat_list_variable, true_treespec) if true_treespec else flat_variable