import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (

        Tests that the google_matrix doesn't change except for the dangling
        nodes.
        