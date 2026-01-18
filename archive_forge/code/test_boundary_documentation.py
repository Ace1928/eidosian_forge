from itertools import combinations
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal
Tests the edge boundary of a multidigraph.