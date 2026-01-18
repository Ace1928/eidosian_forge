import networkx as nx
from networkx.utils.decorators import not_implemented_for, py_random_state
Compute a partitioning of the graphs nodes and the corresponding cut value.

    Use a greedy one exchange strategy to find a locally maximal cut
    and its value, it works by finding the best node (one that gives
    the highest gain to the cut value) to add to the current cut
    and repeats this process until no improvement can be made.

    Parameters
    ----------
    G : networkx Graph
        Graph to find a maximum cut for.

    initial_cut : set
        Cut to use as a starting point. If not supplied the algorithm
        starts with an empty cut.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    cut_value : scalar
        Value of the maximum cut.

    partition : pair of node sets
        A partitioning of the nodes that defines a maximum cut.
    