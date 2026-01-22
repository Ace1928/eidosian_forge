import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import torch
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper
@add_docstring()
class ChangeTrueDivToMulByInverse(ReversibleTransformation):
    """
    Transformation that changes truediv nodes to multiplication by the inverse nodes when the denominator is static.
    For example, that is sometimes the case for the scaling factor in attention layers.
    """
    preserves_computation = True

    def transform(self, graph_module: 'GraphModule') -> 'GraphModule':
        graph = graph_module.graph
        for node in graph.nodes:
            if node.op == 'call_function' and node.target == operator.truediv:
                x, y = node.args
                if not isinstance(y, torch.fx.Node):
                    node.target = operator.mul
                    node.args = (x, 1 / y)
                    self.mark_as_transformed(node)
        return graph_module

    def reverse(self, graph_module: 'GraphModule') -> 'GraphModule':
        for node in self.get_transformed_nodes(graph_module):
            node.target = operator.truediv
            x, y = node.args
            node.args = (x, 1 / y)
            self.mark_as_restored(node)
        return graph_module