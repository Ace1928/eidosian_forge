from __future__ import annotations
import itertools
import operator
from typing import TYPE_CHECKING
import networkx as nx
import numpy as np
from monty.json import MSONable
class SimpleGraphCycle(MSONable):
    """
    Class used to describe a cycle in a simple graph (graph without multiple edges).

    Note that the convention used here is the networkx convention for which simple graphs allow
    to have self-loops in a simple graph.
    No simple graph cycle with two nodes is possible in a simple graph. The graph will not
    be validated if validate is set to False.
    By default, the "ordered" parameter is None, in which case the SimpleGraphCycle will be ordered.
    If the user explicitly sets ordered to False, the SimpleGraphCycle will not be ordered.
    """

    def __init__(self, nodes, validate=True, ordered=None):
        """
        Args:
            nodes:
            validate:
            ordered:
        """
        self.nodes = tuple(nodes)
        if validate:
            self.validate()
        if ordered is not None:
            self.ordered = ordered
        else:
            self.order()

    def _is_valid(self, check_strict_ordering=False):
        """Check if a SimpleGraphCycle is valid.

        This method checks :
        - that there are no duplicate nodes,
        - that there are either 1 or more than 2 nodes

        Returns:
            bool: True if the SimpleGraphCycle is valid.
        """
        if len(self.nodes) == 1:
            return (True, '')
        if len(self.nodes) == 2:
            return (False, 'Simple graph cycle with 2 nodes is not valid.')
        if len(self.nodes) == 0:
            return (False, 'Empty cycle is not valid.')
        if len(self.nodes) != len(set(self.nodes)):
            return (False, 'Duplicate nodes.')
        if check_strict_ordering:
            try:
                sorted_nodes = sorted(self.nodes)
            except TypeError as te:
                msg = te.args[0]
                if "'<' not supported between instances of" in msg:
                    return (False, 'The nodes are not sortable.')
                raise
            res = all((i < j for i, j in zip(sorted_nodes, sorted_nodes[1:])))
            if not res:
                return (False, 'The list of nodes in the cycle cannot be strictly ordered.')
        return (True, '')

    def validate(self, check_strict_ordering=False):
        """
        Args:
            check_strict_ordering:
        """
        is_valid, msg = self._is_valid(check_strict_ordering=check_strict_ordering)
        if not is_valid:
            raise ValueError(f'SimpleGraphCycle is not valid : {msg}')

    def order(self, raise_on_fail=True):
        """Orders the SimpleGraphCycle.

        The ordering is performed such that the first node is the "lowest" one and the
        second node is the lowest one of the two neighbor nodes of the first node. If
        raise_on_fail is set to True a RuntimeError will be raised if the ordering fails.

        Args:
            raise_on_fail (bool): If set to True, will raise a RuntimeError if the ordering fails.
        """
        try:
            self.validate(check_strict_ordering=True)
        except ValueError as ve:
            msg = ve.args[0]
            if 'SimpleGraphCycle is not valid :' in msg and (not raise_on_fail):
                self.ordered = False
                return
            raise
        if len(self.nodes) == 1:
            self.ordered = True
            return
        node_classes = {n.__class__ for n in self.nodes}
        if len(node_classes) > 1:
            if raise_on_fail:
                raise ValueError('Could not order simple graph cycle as the nodes are of different classes.')
            self.ordered = False
            return
        min_index, _min_node = min(enumerate(self.nodes), key=operator.itemgetter(1))
        reverse = self.nodes[(min_index - 1) % len(self.nodes)] < self.nodes[(min_index + 1) % len(self.nodes)]
        if reverse:
            self.nodes = self.nodes[min_index::-1] + self.nodes[:min_index:-1]
        else:
            self.nodes = self.nodes[min_index:] + self.nodes[:min_index]
        self.ordered = True

    def __hash__(self) -> int:
        return len(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        out = ['Simple cycle with nodes :']
        out.extend([str(node) for node in self.nodes])
        return '\n'.join(out)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimpleGraphCycle):
            return NotImplemented
        if not self.ordered or not other.ordered:
            raise RuntimeError('Simple cycles should be ordered in order to be compared.')
        return self.nodes == other.nodes

    @classmethod
    def from_edges(cls, edges, edges_are_ordered: bool=True) -> Self:
        """Constructs SimpleGraphCycle from a list edges.

        By default, the edges list is supposed to be ordered as it will be
        much faster to construct the cycle. If edges_are_ordered is set to
        False, the code will automatically try to find the corresponding edge
        order in the list.
        """
        if edges_are_ordered:
            nodes = [edge[0] for edge in edges]
            if not all((e1e2[0][1] == e1e2[1][0] for e1e2 in zip(edges, edges[1:]))) or edges[-1][1] != edges[0][0]:
                raise ValueError('Could not construct a cycle from edges.')
        else:
            remaining_edges = list(edges)
            nodes = list(remaining_edges.pop())
            while remaining_edges:
                prev_node = nodes[-1]
                for ie, e in enumerate(remaining_edges):
                    if prev_node == e[0]:
                        remaining_edges.pop(ie)
                        nodes.append(e[1])
                        break
                    if prev_node == e[1]:
                        remaining_edges.pop(ie)
                        nodes.append(e[0])
                        break
                else:
                    raise ValueError('Could not construct a cycle from edges.')
            if nodes[0] != nodes[-1]:
                raise ValueError('Could not construct a cycle from edges.')
            nodes.pop()
        return cls(nodes)

    def as_dict(self) -> dict:
        """MSONable dict"""
        dct = MSONable.as_dict(self)
        dct['nodes'] = list(dct['nodes'])
        return dct

    @classmethod
    def from_dict(cls, dct: dict, validate: bool=False) -> Self:
        """
        Serialize from dict.

        Args:
            dct (dict): Dict representation.
            validate: If True, will validate the cycle.
        """
        return cls(nodes=dct['nodes'], validate=validate, ordered=dct['ordered'])