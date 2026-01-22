from ``u`` to ``v`` that is not blocked. That means, there is an open
from collections import deque
from itertools import chain
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for
@not_implemented_for('undirected')
def _reachable(G, x, a, z):
    """Modified Bayes-Ball algorithm for finding d-connected nodes.

    Find all nodes in `a` that are d-connected to those in `x` by
    those in `z`. This is an implementation of the function
    `REACHABLE` in [1]_ (which is itself a modification of the
    Bayes-Ball algorithm [2]_) when restricted to DAGs.

    Parameters
    ----------
    G : nx.DiGraph
        A NetworkX DAG.
    x : node | set
        A node in the DAG, or a set of nodes.
    a : node | set
        A (set of) node(s) in the DAG containing the ancestors of `x`.
    z : node | set
        The node or set of nodes conditioned on when checking d-connectedness.

    Returns
    -------
    w : set
        The closure of `x` in `a` with respect to d-connectedness
        given `z`.

    References
    ----------
    .. [1] van der Zander, Benito, and Maciej Liśkiewicz. "Finding
        minimal d-separators in linear time and applications." In
        Uncertainty in Artificial Intelligence, pp. 637-647. PMLR, 2020.

    .. [2] Shachter, Ross D. "Bayes-ball: The rational pastime
       (for determining irrelevance and requisite information in
       belief networks and influence diagrams)." In Proceedings of the
       Fourteenth Conference on Uncertainty in Artificial Intelligence
       (UAI), (pp. 480–487). 1998.
    """

    def _pass(e, v, f, n):
        """Whether a ball entering node `v` along edge `e` passes to `n` along `f`.

        Boolean function defined on page 6 of [1]_.

        Parameters
        ----------
        e : bool
            Directed edge by which the ball got to node `v`; `True` iff directed into `v`.
        v : node
            Node where the ball is.
        f : bool
            Directed edge connecting nodes `v` and `n`; `True` iff directed `n`.
        n : node
            Checking whether the ball passes to this node.

        Returns
        -------
        b : bool
            Whether the ball passes or not.

        References
        ----------
        .. [1] van der Zander, Benito, and Maciej Liśkiewicz. "Finding
           minimal d-separators in linear time and applications." In
           Uncertainty in Artificial Intelligence, pp. 637-647. PMLR, 2020.
        """
        is_element_of_A = n in a
        collider_if_in_Z = v not in z or (e and (not f))
        return is_element_of_A and collider_if_in_Z
    queue = deque([])
    for node in x:
        if bool(G.pred[node]):
            queue.append((True, node))
        if bool(G.succ[node]):
            queue.append((False, node))
    processed = queue.copy()
    while any(queue):
        e, v = queue.popleft()
        preds = ((False, n) for n in G.pred[v])
        succs = ((True, n) for n in G.succ[v])
        f_n_pairs = chain(preds, succs)
        for f, n in f_n_pairs:
            if (f, n) not in processed and _pass(e, v, f, n):
                queue.append((f, n))
                processed.append((f, n))
    return {w for _, w in processed}