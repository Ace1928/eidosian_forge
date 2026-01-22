from __future__ import annotations
import itertools
import operator
from typing import TYPE_CHECKING
import networkx as nx
import numpy as np
from monty.json import MSONable
def _is_valid(self, check_strict_ordering=False):
    """Check if a MultiGraphCycle is valid.

        This method checks that:
        1. there are no duplicate nodes,
        2. there are either 1 or more than 2 nodes

        Returns:
            bool: True if the SimpleGraphCycle is valid.
        """
    if len(self.nodes) != len(self.edge_indices):
        return (False, 'Number of nodes different from number of edge indices.')
    if len(self.nodes) == 0:
        return (False, 'Empty cycle is not valid.')
    if len(self.nodes) != len(set(self.nodes)):
        return (False, 'Duplicate nodes.')
    if len(self.nodes) == 2 and self.edge_indices[0] == self.edge_indices[1]:
        return (False, 'Cycles with two nodes cannot use the same edge for the cycle.')
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