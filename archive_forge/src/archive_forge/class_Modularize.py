from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
class Modularize(_pass.Transform):
    """Transforms a flattened `fx.GraphModule` into a modular structure.

    In the flattened `fx.GraphModule`, each `nn.Module` forward call has been traced as
    a sequence of `fx.Node`s. All these `fx.Node`s are flattened and reside in the same
    `fx.GraphModule`.

    This pass generates a new `fx.GraphModule`. It groups the flattened `fx.Node`s that belong
    to the same `nn.Module` forward call into a sub `fx.GraphModule`. It then replaces the
    sequence of flattened `fx.Node`s with a single `call_module` node, which is linked with
    the sub `fx.GraphModule` by `node.target`. The sub `fx.GraphModule` is registered as a
    submodule of the new `fx.GraphModule`.

    The process is done based on information from the `nn_module_stack` metadata of each node, i.e.
    `node.meta["nn_module_stack"]`. For more implementation details, see [NOTE: Modularize Pass Implementation].

    An fx submodule under this context can typically be interpreted in three different ways:

        1. As an embodiment of an nn.Module class, which is considered stateless.
        Its execution path can vary depending on the configuration of module initialization,
        which should also be part of the inputs.

        2. As a representation of an nn.Module instance. It maintains the state initialized in the module.
        The execution path can vary based on actual input data.

        3. As a captured call of an nn.Module instance, where the execution path
        is set.

    The generality decreases along this list. Within the scope of this function, the pass
    creates fx submodules according to the third interpretation.

    The first interpretation is the most general case. It requires complex analysis and additional
    metadata and code information to construct its general form. Consider an example nn.Module
    that generates arbitrary submodules based on an initialization configuration file. It's impractical
    to extract this logic for the generated fx submodule to function with arbitrary configuration.

    The second interpretation demands less analysis and is sturdier than the
    first. In most use cases, it's equivalent to the third. It only differs in exceptional situations
    where a complex nn.Module instance is called multiple times, each with a different set of inputs
    leading to a unique execution branching path.

    The third interpretation is the most specific scenario. It necessitates the minimum
    analysis and creates the most stable representation. The drawback is that it
    generates more redundancy than the other two methods. If needed, a subsequent post-processing
    pass can be applied to consolidate completely identical functions and reduce duplication.

    ### Known constraints
    Two successive calls to the same module instance will be conflated. They are indistinguishable.
    This is due to limitations of the current fx metadata "nn_module_stack".

    [NOTE: Modularize pass ordering]
    This pass groups fx nodes into subgraphs that reside within the `call_module` fx node.
    Other fx passes (including some outside the exporter) might not recognize `call_module`.
    They may assume that all nodes are flattened. Hence it is recommended to invoke this pass
    as the last pre onnx export fx pass. If not for this consideration, this operation could
    potentially be relocated anywhere earlier in the pipeline.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> import torch
        >>> from torch.onnx._internal.fx import passes
        >>> from torch.onnx._internal.diagnostics import infra
        >>>
        >>> class CustomModule(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.embedding = torch.nn.Embedding(10, 32)
        >>>         self.relu = torch.nn.ReLU()
        >>>
        >>>     def forward(self, x):
        >>>         out = self.embedding(x)
        >>>         out = self.relu(out)
        >>>         return out
        >>>
        >>> class TestModule(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.layer = CustomModule()
        >>>         self.linear = torch.nn.Linear(32, 10)
        >>>
        >>>     def forward(self, x):
        >>>         out = self.layer(x)
        >>>         out = self.linear(out)
        >>>         return out
        >>>
        >>> gm, _ = torch._dynamo.export(TestModule(), aten_graph=True)(torch.tensor([0, 1, 2]))
        >>> gm.print_readable()

        >>> gm = passes.Modularize(infra.DiagnosticContext("test_context", "1.0"), gm).run()
        >>> gm.print_readable()

    """

    @_beartype.beartype
    def _run(self) -> torch.fx.GraphModule:
        self.module.graph.eliminate_dead_code()
        reference_module = torch.fx.GraphModule(self.module, self.module.graph)
        root_module_node = _ModuleNode(reference_module, _ModuleStackMeta(None))
        for fx_node in self.module.graph.nodes:
            root_module_node.add_leaf_node(_LeafNode(fx_node))
        return root_module_node.build_module({})