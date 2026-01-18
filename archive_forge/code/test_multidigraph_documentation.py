from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_multigraph import BaseMultiGraphTester
from .test_multigraph import TestEdgeSubgraph as _TestMultiGraphEdgeSubgraph
from .test_multigraph import TestMultiGraph as _TestMultiGraph
Unit tests for the :meth:`MultiDiGraph.edge_subgraph` method.