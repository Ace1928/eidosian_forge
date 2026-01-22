import torch
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map_only
from torch.utils import _pytree as pytree
from torch.multiprocessing.reductions import StorageWeakRef
import _operator
from enum import Enum
import itertools
from typing import Set, Dict
from collections import defaultdict
@compatibility(is_backward_compatible=False)
class _FunctionalizationMetadataProp(torch.fx.Interpreter):

    def run_node(self, node: Node):
        self.node_counter += 1
        result = super().run_node(node)
        node.meta['fake_result'] = result
        node.meta['node_idx'] = self.node_counter
        node_args = node.args
        if node.target is torch.ops.aten.copy_.default:
            node_args = node_args[1:]
        if node.op == 'call_function':
            view_type = _get_view_type(node.target)
            if view_type == _ViewType.SingleOutputView:
                assert isinstance(node.args[0], Node)
                node.meta['view_of'] = node.args[0]
            elif view_type == _ViewType.MultiOutputView:
                self.multi_output_view_nodes[node] = node.args[0]
            elif node.target is _operator.getitem:
                list_arg = node.args[0]
                maybe_base_of_view = self.multi_output_view_nodes.get(list_arg, None)
                if maybe_base_of_view is not None:
                    assert isinstance(maybe_base_of_view, Node)
                    node.meta['view_of'] = maybe_base_of_view
        if 'view_of' in node.meta:
            assert isinstance(node.meta['fake_result'], FakeTensor)
            assert isinstance(node.meta['view_of'].meta['fake_result'], FakeTensor)
            view_storage = StorageWeakRef(node.meta['fake_result']._typed_storage())
            base_storage = StorageWeakRef(node.meta['view_of'].meta['fake_result']._typed_storage())
            assert view_storage == base_storage
        return result

    def propagate(self, *args):
        self.multi_output_view_nodes = {}
        self.node_counter = -1
        with FakeTensorMode() as mode:
            fake_args = [mode.from_tensor(a) for a in args]
            return super().run(*fake_args)