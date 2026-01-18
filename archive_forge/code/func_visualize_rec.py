import time
from collections import defaultdict
from functools import partial
from typing import DefaultDict
import torch
def visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it=None):
    """Recursive part of visualize (basically skips setting up the input and output nodes)."""

    def inline_graph(subgraph, name, node):
        rec_value_map = {inp.unique(): value_map[val.unique()] for inp, val in zip(subgraph.inputs(), node.inputs())}
        visualize_rec(graph=subgraph, value_map=rec_value_map, name_prefix=name, pb_graph=pb_graph)
        for out, val in zip(subgraph.outputs(), node.outputs()):
            value_map[val.unique()] = rec_value_map[out.unique()]
    op_id_counter: DefaultDict[str, int] = defaultdict(int)

    def name_for(node):
        kind = node.kind()[node.kind().index('::') + 2:]
        op_id_counter[kind] += 1
        return (kind, name_prefix + kind + '_' + str(op_id_counter[kind]))

    def add_fusion_group(node):
        op, name = name_for(node)
        inline_graph(node.g('Subgraph'), name + '/', node)

    def add_graph_executor(node):
        op, name = name_for(node)
        if executors_it is None:
            add_node(node)
        else:
            ge = next(executors_it)
            visualize_graph_executor(ge, name + '/', pb_graph, partial(inline_graph, node=node))

    def add_node(node):
        if node.kind() == 'prim::FusionGroup':
            return add_fusion_group(node)
        elif node.kind() == 'prim::GraphExecutor':
            return add_graph_executor(node)
        op, name = name_for(node)
        pb_node = pb_graph.node.add(op=op, name=name)
        for value in node.inputs():
            pb_node.input.append(value_map[value.unique()])
        for i, value in enumerate(node.outputs()):
            value_map[value.unique()] = name + ':' + str(i)
    for node in graph.nodes():
        add_node(node)