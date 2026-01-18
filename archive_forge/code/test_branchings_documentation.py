import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition

    Tests the arborescence iterator with three included edges and three excluded
    in the initial partition.

    A brute force method similar to the one used in the above tests found that
    there are 16 arborescences which contain the included edges and not the
    excluded edges.
    