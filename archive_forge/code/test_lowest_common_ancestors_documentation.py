from itertools import chain, combinations, product
import pytest
import networkx as nx
Checks if d1 and d2 contain the same pairs and
        have a node at the same distance from root for each.
        If G is None use self.DG.