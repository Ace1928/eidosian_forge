import heapq
import networkx as nx
Returns True if some directed graph can realize the in- and out-degree
    sequences.

    Parameters
    ----------
    in_sequence : list or iterable container
        A sequence of integer node in-degrees

    out_sequence : list or iterable container
        A sequence of integer node out-degrees

    Returns
    -------
    valid : bool
      True if in and out-sequences are digraphic False if not.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])
    >>> in_seq = (d for n, d in G.in_degree())
    >>> out_seq = (d for n, d in G.out_degree())
    >>> nx.is_digraphical(in_seq, out_seq)
    True

    To test a non-digraphical scenario:
    >>> in_seq_list = [d for n, d in G.in_degree()]
    >>> in_seq_list[-1] += 1
    >>> nx.is_digraphical(in_seq_list, out_seq)
    False

    Notes
    -----
    This algorithm is from Kleitman and Wang [1]_.
    The worst case runtime is $O(s \times \log n)$ where $s$ and $n$ are the
    sum and length of the sequences respectively.

    References
    ----------
    .. [1] D.J. Kleitman and D.L. Wang
       Algorithms for Constructing Graphs and Digraphs with Given Valences
       and Factors, Discrete Mathematics, 6(1), pp. 79-88 (1973)
    