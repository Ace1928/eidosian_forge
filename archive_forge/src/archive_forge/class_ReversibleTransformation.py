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
@add_docstring(add_example=False)
class ReversibleTransformation(Transformation):
    """
    A torch.fx graph transformation that is reversible.

    It must implement the [`~optimum.fx.optimization.ReversibleTransformation.transform`] and
    [`~optimum.fx.optimization.ReversibleTransformation.reverse`] methods, and be used as a callable.
    """

    @abstractmethod
    def reverse(self, graph_module: 'GraphModule') -> 'GraphModule':
        """
        Args:
            graph_module (`torch.fx.GraphModule`):
                The module to transform.

        Returns:
            `torch.fx.GraphModule`:
                The reverse transformed module.
        """
        raise NotImplementedError('The reverse transform method needs to be implemented.')

    def __call__(self, graph_module: 'GraphModule', lint_and_recompile: bool=True, reverse: bool=False) -> 'GraphModule':
        """
        Args:
            graph_module (`torch.fx.GraphModule`):
                The module to transform.
            lint_and_recompile (`bool`, defaults to `True`):
                Whether the transformed module should be linted and recompiled.
                This can be set to `False` when chaining transformations together to perform this operation only once.
            reverse (`bool`, defaults to `False`):
                If `True`, the reverse transformation is performed.

        Returns:
            `torch.fx.GraphModule`:
                The transformed module.

        """
        func = self.transform if not reverse else self.reverse
        graph_module = func(graph_module)
        if lint_and_recompile:
            graph_module.graph.lint()
            graph_module.recompile()
        return graph_module

    def mark_as_restored(self, node: 'Node'):
        """
        Marks a node as restored back to its original state.

        Args:
            node (`torch.fx.Node`):
                The node to mark as restored.
        """
        node_transformations = getattr(node, 'transformations', set())
        if self.signature not in node_transformations:
            raise ValueError('The node was not transformed by this transformation.')
        node_transformations.remove(self.signature)