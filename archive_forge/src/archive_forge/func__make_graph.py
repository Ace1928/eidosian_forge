from __future__ import annotations
import logging # isort:skip
from itertools import permutations
from typing import TYPE_CHECKING
from bokeh.core.properties import UnsetValueError
from bokeh.layouts import column
from bokeh.models import (
def _make_graph(self, M: Model):
    """ Builds a networkx DiGraph() G from the model M.


        G.nodes are the submodels of M, with node attributes
            - "model" giving the class name of of the submodel
            - "id" giving the id of the submodel

        An edge runs from a to b if the submodel b belongs to an attribute of
            a

        Args:
            A bokeh model M

        """
    import networkx as nx

    def test_condition(s: Model, y: str, H: Model) -> bool:
        answer1: bool = False
        answer2: bool = False
        answer3: bool = False
        try:
            answer1 = s in getattr(H, y)
        except (TypeError, UnsetValueError):
            pass
        try:
            answer2 = s == getattr(H, y)
        except (TypeError, UnsetValueError):
            pass
        try:
            answer3 = s in getattr(H, y).values()
        except (AttributeError, ValueError, UnsetValueError):
            pass
        return answer1 | answer2 | answer3
    K = nx.DiGraph()
    T: dict[ID, set[ID]] = {}
    for m in M.references():
        T[m.id] = {y.id for y in m.references()}
    K.add_nodes_from([(x, {'model': M.select_one({'id': x}).__class__.__name__}) for x in T])
    E = [(y, x) for x, y in permutations(T, 2) if T[x] <= T[y]]
    K.add_edges_from(E)
    dead_edges = []
    for id in K.nodes:
        H = M.select_one({'id': id})
        for x in K.neighbors(id):
            s = H.select_one({'id': x})
            keep_edge = False
            for y in H.properties():
                if test_condition(s, y, H):
                    keep_edge = True
            if not keep_edge:
                dead_edges.append((id, x))
    K.remove_edges_from(dead_edges)
    return K