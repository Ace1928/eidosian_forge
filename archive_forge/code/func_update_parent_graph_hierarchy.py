from pyparsing import (
import pydot
def update_parent_graph_hierarchy(g, parent_graph=None, level=0):
    if parent_graph is None:
        parent_graph = g
    for key_name in ('edges',):
        if isinstance(g, pydot.frozendict):
            item_dict = g
        else:
            item_dict = g.obj_dict
        if key_name not in item_dict:
            continue
        for key, objs in item_dict[key_name].items():
            for obj in objs:
                if 'parent_graph' in obj and obj['parent_graph'].get_parent_graph() == g:
                    if obj['parent_graph'] is g:
                        pass
                    else:
                        obj['parent_graph'].set_parent_graph(parent_graph)
                if key_name == 'edges' and len(key) == 2:
                    for idx, vertex in enumerate(obj['points']):
                        if isinstance(vertex, (pydot.Graph, pydot.Subgraph, pydot.Cluster)):
                            vertex.set_parent_graph(parent_graph)
                        if isinstance(vertex, pydot.frozendict):
                            if vertex['parent_graph'] is g:
                                pass
                            else:
                                vertex['parent_graph'].set_parent_graph(parent_graph)