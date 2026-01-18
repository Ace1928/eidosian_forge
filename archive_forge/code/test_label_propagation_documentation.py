from itertools import chain, combinations
import pytest
import networkx as nx
Checks that the communities computed from the given graph ``G``
        using the :func:`~networkx.asyn_lpa_communities` function match
        the set of nodes given in ``expected``.

        ``expected`` must be a :class:`set` of :class:`frozenset`
        instances, each element of which is a node in the graph.

        