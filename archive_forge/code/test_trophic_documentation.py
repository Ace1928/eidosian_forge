import pytest
import networkx as nx
Should fail to compute if there are any parts of the graph which are not
    reachable from any basal node (with in-degree zero).
    