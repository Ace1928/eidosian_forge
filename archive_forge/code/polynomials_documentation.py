from collections import deque
import networkx as nx
from networkx.utils import not_implemented_for
Returns the chromatic polynomial of `G`

    This function computes the chromatic polynomial via an iterative version of
    the deletion-contraction algorithm.

    The chromatic polynomial `X_G(x)` is a fundamental graph polynomial
    invariant in one variable. Evaluating `X_G(k)` for an natural number `k`
    enumerates the proper k-colorings of `G`.

    There are several equivalent definitions; here are three:

    Def 1 (explicit formula):
    For `G` an undirected graph, `c(G)` the number of connected components of
    `G`, `E` the edge set of `G`, and `G(S)` the spanning subgraph of `G` with
    edge set `S` [1]_:

    .. math::

        X_G(x) = \sum_{S \subseteq E} (-1)^{|S|} x^{c(G(S))}


    Def 2 (interpolating polynomial):
    For `G` an undirected graph, `n(G)` the number of vertices of `G`, `k_0 = 0`,
    and `k_i` the number of distinct ways to color the vertices of `G` with `i`
    unique colors (for `i` a natural number at most `n(G)`), `X_G(x)` is the
    unique Lagrange interpolating polynomial of degree `n(G)` through the points
    `(0, k_0), (1, k_1), \dots, (n(G), k_{n(G)})` [2]_.


    Def 3 (chromatic recurrence):
    For `G` an undirected graph, `G-e` the graph obtained from `G` by deleting
    edge `e`, `G/e` the graph obtained from `G` by contracting edge `e`, `n(G)`
    the number of vertices of `G`, and `e(G)` the number of edges of `G` [3]_:

    .. math::
        X_G(x) = \begin{cases}
    	   x^{n(G)}, & \text{if $e(G)=0$} \\
           X_{G-e}(x) - X_{G/e}(x), & \text{otherwise, for an arbitrary edge $e$}
        \end{cases}

    This formulation is also known as the Fundamental Reduction Theorem [4]_.


    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    instance of `sympy.core.add.Add`
        A Sympy expression representing the chromatic polynomial for `G`.

    Examples
    --------
    >>> C = nx.cycle_graph(5)
    >>> nx.chromatic_polynomial(C)
    x**5 - 5*x**4 + 10*x**3 - 10*x**2 + 4*x

    >>> G = nx.complete_graph(4)
    >>> nx.chromatic_polynomial(G)
    x**4 - 6*x**3 + 11*x**2 - 6*x

    Notes
    -----
    Interpretation of the coefficients is discussed in [5]_. Several special
    cases are listed in [2]_.

    The chromatic polynomial is a specialization of the Tutte polynomial; in
    particular, ``X_G(x) = T_G(x, 0)`` [6]_.

    The chromatic polynomial may take negative arguments, though evaluations
    may not have chromatic interpretations. For instance, ``X_G(-1)`` enumerates
    the acyclic orientations of `G` [7]_.

    References
    ----------
    .. [1] D. B. West,
       "Introduction to Graph Theory," p. 222
    .. [2] E. W. Weisstein
       "Chromatic Polynomial"
       MathWorld--A Wolfram Web Resource
       https://mathworld.wolfram.com/ChromaticPolynomial.html
    .. [3] D. B. West,
       "Introduction to Graph Theory," p. 221
    .. [4] J. Zhang, J. Goodall,
       "An Introduction to Chromatic Polynomials"
       https://math.mit.edu/~apost/courses/18.204_2018/Julie_Zhang_paper.pdf
    .. [5] R. C. Read,
       "An Introduction to Chromatic Polynomials"
       Journal of Combinatorial Theory, 1968
       https://math.berkeley.edu/~mrklug/ReadChromatic.pdf
    .. [6] W. T. Tutte,
       "Graph-polynomials"
       Advances in Applied Mathematics, 2004
       https://www.sciencedirect.com/science/article/pii/S0196885803000411
    .. [7] R. P. Stanley,
       "Acyclic orientations of graphs"
       Discrete Mathematics, 2006
       https://math.mit.edu/~rstan/pubs/pubfiles/18.pdf
    