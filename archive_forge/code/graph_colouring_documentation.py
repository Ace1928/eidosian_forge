import numpy as np
Performs graph-colouring using the Recursive Largest Degree First heuristic. Often yields a
    lower chromatic number than Largest Degree First, but takes longer (runtime is cubic in number
    of vertices).

    Args:
        binary_observables (array[int]): the set of Pauli words represented by a column matrix of
            the Pauli words in binary vector represenation
        adj (array[int]): the adjacency matrix of the Pauli graph

    Returns:
        dict(int, list[array[int]]): keys correspond to colours (labelled by integers) and values
        are lists of Pauli words of the same colour in binary vector representation

    **Example**

    >>> binary_observables = np.array([[1., 1., 0.],
    ... [1., 0., 0.],
    ... [0., 0., 1.],
    ... [1., 0., 1.]])
    >>> adj = np.array([[0., 0., 1.],
    ... [0., 0., 1.],
    ... [1., 1., 0.]])
    >>> recursive_largest_first(binary_observables, adj)
    {1: [array([0., 0., 1.])], 2: [array([1., 1., 0.]), array([1., 0., 0.])]}
    