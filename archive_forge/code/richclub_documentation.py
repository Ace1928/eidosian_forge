from itertools import accumulate
import networkx as nx
from networkx.utils import not_implemented_for
Returns the rich-club coefficient for each degree in the graph
    `G`.

    `G` is an undirected graph without multiedges.

    Returns a dictionary mapping degree to rich-club coefficient for
    that degree.

    