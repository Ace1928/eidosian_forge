from collections import deque
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
from networkx.utils import pairwise
Build a path using DFS starting from the sink