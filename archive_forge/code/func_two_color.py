import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def two_color(graph):
    """Compute a two-coloring of a directed graph

    If a two coloring is not possible for the input graph (meaning it is not
    bipartite), ``None`` is returned.

    :param graph: The graph to find the coloring for
    :returns: If a coloring is possible return a dictionary of node indices to the color as an integer (0 or 1)
    :rtype: dict
    """