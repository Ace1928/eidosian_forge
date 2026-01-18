import time
from collections import defaultdict
from functools import partial
from typing import DefaultDict
import torch
def visualize_graph_executor(state, name_prefix, pb_graph, inline_graph):
    """Append the state of a given GraphExecutor to the graph protobuf.

    Args:
        state (GraphExecutor or GraphExecutorState): GraphExecutor to display.
        name_prefix (str): Name prefix of the containing subgraph.
        pb_graph (GraphDef): graph to append to.
        inline_graph (Callable): a function that handles setting up a value_map,
            so that some graphs in here can be inlined. This is necessary, because
            this will simply be `visualize` for the top-level GraphExecutor,
            or `inline_graph` for all nested ones.

            The signature should look like (Graph, name_prefix) -> ().
            It will be called exactly once.

    The strategy is to embed all different configurations as independent subgraphs,
    while inlining the original graph as the one that actually produces the values.
    """
    if state.autograd_fallback_graph is not None:
        visualize(graph=state.autograd_fallback_graph, name_prefix=name_prefix + 'autograd_fallback/', pb_graph=pb_graph, executors_it=iter(state.autograd_fallback.executors()))
    for i, (arg_spec, plan) in enumerate(state.execution_plans.items()):
        subgraph_name = name_prefix + f'plan{i}/'
        input_kinds = pb_graph.node.add(op='INPUT_KIND', name=subgraph_name)
        input_kinds.attr['inputs'].s = repr(arg_spec).encode('ascii')
        visualize(plan.graph, subgraph_name, pb_graph, iter(plan.code.executors()))
        if plan.grad_executor is not None:
            grad_subgraph_name = subgraph_name + 'grad/'
            visualize(plan.grad_executor, grad_subgraph_name, pb_graph)
    return inline_graph(state.graph, name_prefix + 'original/')