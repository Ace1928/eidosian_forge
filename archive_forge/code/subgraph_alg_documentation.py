import networkx as nx
from networkx.utils import not_implemented_for
Returns the Estrada index of a the graph G.

    The Estrada Index is a topological index of folding or 3D "compactness" ([1]_).

    Parameters
    ----------
    G: graph

    Returns
    -------
    estrada index: float

    Raises
    ------
    NetworkXError
        If the graph is not undirected and simple.

    Notes
    -----
    Let `G=(V,E)` be a simple undirected graph with `n` nodes  and let
    `\lambda_{1}\leq\lambda_{2}\leq\cdots\lambda_{n}`
    be a non-increasing ordering of the eigenvalues of its adjacency
    matrix `A`. The Estrada index is ([1]_, [2]_)

    .. math::
        EE(G)=\sum_{j=1}^n e^{\lambda _j}.

    References
    ----------
    .. [1] E. Estrada, "Characterization of 3D molecular structure",
       Chem. Phys. Lett. 319, 713 (2000).
       https://doi.org/10.1016/S0009-2614(00)00158-5
    .. [2] José Antonio de la Peñaa, Ivan Gutman, Juan Rada,
       "Estimating the Estrada index",
       Linear Algebra and its Applications. 427, 1 (2007).
       https://doi.org/10.1016/j.laa.2007.06.020

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])
    >>> ei = nx.estrada_index(G)
    >>> print(f"{ei:0.5}")
    20.55
    