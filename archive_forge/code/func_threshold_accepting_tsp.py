important in operations research and theoretical computer science.
import math
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.utils import not_implemented_for, pairwise, py_random_state
@py_random_state(9)
@nx._dispatch(edge_attrs='weight')
def threshold_accepting_tsp(G, init_cycle, weight='weight', source=None, threshold=1, move='1-1', max_iterations=10, N_inner=100, alpha=0.1, seed=None):
    """Returns an approximate solution to the traveling salesman problem.

    This function uses threshold accepting methods to approximate the minimal cost
    cycle through the nodes. Starting from a suboptimal solution, threshold
    accepting methods perturb that solution, accepting any changes that make
    the solution no worse than increasing by a threshold amount. Improvements
    in cost are accepted, but so are changes leading to small increases in cost.
    This allows the solution to leave suboptimal local minima in solution space.
    The threshold is decreased slowly as iterations proceed helping to ensure
    an optimum. In summary, the function returns a cycle starting at `source`
    for which the total cost is minimized.

    Parameters
    ----------
    G : Graph
        `G` should be a complete weighted graph.
        The distance between all pairs of nodes should be included.

    init_cycle : list or "greedy"
        The initial solution (a cycle through all nodes returning to the start).
        This argument has no default to make you think about it.
        If "greedy", use `greedy_tsp(G, weight)`.
        Other common starting cycles are `list(G) + [next(iter(G))]` or the final
        result of `simulated_annealing_tsp` when doing `threshold_accepting_tsp`.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    source : node, optional (default: first node in list(G))
        Starting node.  If None, defaults to ``next(iter(G))``

    threshold : int, optional (default=1)
        The algorithm's threshold parameter. It represents the initial
        threshold's value

    move : "1-1" or "1-0" or function, optional (default="1-1")
        Indicator of what move to use when finding new trial solutions.
        Strings indicate two special built-in moves:

        - "1-1": 1-1 exchange which transposes the position
          of two elements of the current solution.
          The function called is :func:`swap_two_nodes`.
          For example if we apply 1-1 exchange in the solution
          ``A = [3, 2, 1, 4, 3]``
          we can get the following by the transposition of 1 and 4 elements:
          ``A' = [3, 2, 4, 1, 3]``
        - "1-0": 1-0 exchange which moves an node in the solution
          to a new position.
          The function called is :func:`move_one_node`.
          For example if we apply 1-0 exchange in the solution
          ``A = [3, 2, 1, 4, 3]``
          we can transfer the fourth element to the second position:
          ``A' = [3, 4, 2, 1, 3]``

        You may provide your own functions to enact a move from
        one solution to a neighbor solution. The function must take
        the solution as input along with a `seed` input to control
        random number generation (see the `seed` input here).
        Your function should maintain the solution as a cycle with
        equal first and last node and all others appearing once.
        Your function should return the new solution.

    max_iterations : int, optional (default=10)
        Declared done when this number of consecutive iterations of
        the outer loop occurs without any change in the best cost solution.

    N_inner : int, optional (default=100)
        The number of iterations of the inner loop.

    alpha : float between (0, 1), optional (default=0.1)
        Percentage of threshold decrease when there is at
        least one acceptance of a neighbor solution.
        If no inner loop moves are accepted the threshold remains unchanged.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    cycle : list of nodes
        Returns the cycle (list of nodes) that a salesman
        can follow to minimize total weight of the trip.

    Raises
    ------
    NetworkXError
        If `G` is not complete the algorithm raises an exception.

    Examples
    --------
    >>> from networkx.algorithms import approximation as approx
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from({
    ...     ("A", "B", 3), ("A", "C", 17), ("A", "D", 14), ("B", "A", 3),
    ...     ("B", "C", 12), ("B", "D", 16), ("C", "A", 13),("C", "B", 12),
    ...     ("C", "D", 4), ("D", "A", 14), ("D", "B", 15), ("D", "C", 2)
    ... })
    >>> cycle = approx.threshold_accepting_tsp(G, "greedy", source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31
    >>> incycle = ["D", "B", "A", "C", "D"]
    >>> cycle = approx.threshold_accepting_tsp(G, incycle, source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31

    Notes
    -----
    Threshold Accepting is a metaheuristic local search algorithm.
    The main characteristic of this algorithm is that it accepts
    even solutions which lead to the increase of the cost in order
    to escape from low quality local optimal solutions.

    This algorithm needs an initial solution. This solution can be
    constructed by a simple greedy algorithm. At every iteration, it
    selects thoughtfully a neighbor solution.
    Consider $c(x)$ cost of current solution and $c(x')$ cost of
    neighbor solution.
    If $c(x') - c(x) <= threshold$ then the neighbor solution becomes the current
    solution for the next iteration, where the threshold is named threshold.

    In comparison to the Simulated Annealing algorithm, the Threshold
    Accepting algorithm does not accept very low quality solutions
    (due to the presence of the threshold value). In the case of
    Simulated Annealing, even a very low quality solution can
    be accepted with probability $p$.

    Time complexity:
    It has a running time $O(m * n * |V|)$ where $m$ and $n$ are the number
    of times the outer and inner loop run respectively.

    For more information and how algorithm is inspired see:
    https://doi.org/10.1016/0021-9991(90)90201-B

    See Also
    --------
    simulated_annealing_tsp

    """
    if move == '1-1':
        move = swap_two_nodes
    elif move == '1-0':
        move = move_one_node
    if init_cycle == 'greedy':
        cycle = greedy_tsp(G, weight=weight, source=source)
        if G.number_of_nodes() == 2:
            return cycle
    else:
        cycle = list(init_cycle)
        if source is None:
            source = cycle[0]
        elif source != cycle[0]:
            raise nx.NetworkXError('source must be first node in init_cycle')
        if cycle[0] != cycle[-1]:
            raise nx.NetworkXError('init_cycle must be a cycle. (return to start)')
        if len(cycle) - 1 != len(G) or len(set(G.nbunch_iter(cycle))) != len(G):
            raise nx.NetworkXError('init_cycle is not all and only nodes.')
        N = len(G) - 1
        if any((len(nbrdict) - (n in nbrdict) != N for n, nbrdict in G.adj.items())):
            raise nx.NetworkXError('G must be a complete graph.')
        if G.number_of_nodes() == 2:
            neighbor = list(G.neighbors(source))[0]
            return [source, neighbor, source]
    cost = sum((G[u][v].get(weight, 1) for u, v in pairwise(cycle)))
    count = 0
    best_cycle = cycle.copy()
    best_cost = cost
    while count <= max_iterations:
        count += 1
        accepted = False
        for i in range(N_inner):
            adj_sol = move(cycle, seed)
            adj_cost = sum((G[u][v].get(weight, 1) for u, v in pairwise(adj_sol)))
            delta = adj_cost - cost
            if delta <= threshold:
                accepted = True
                cycle = adj_sol
                cost = adj_cost
                if cost < best_cost:
                    count = 0
                    best_cycle = cycle.copy()
                    best_cost = cost
        if accepted:
            threshold -= threshold * alpha
    return best_cycle