from pennylane.pauli.utils import diagonalize_qwc_groupings
from .group_observables import group_observables
Partitions then diagonalizes a list of Pauli words, facilitating simultaneous measurement of
    all observables within a partition.

    The input list of observables are partitioned into mutually qubit-wise commuting (QWC) or
    mutually commuting partitions by approximately solving minimum clique cover on a graph where
    each observable represents a vertex. The unitaries which diagonalize the
    partitions are then found. See `arXiv:1907.03358
    <https://arxiv.org/abs/1907.03358>`_ and `arXiv:1907.09386
    <https://arxiv.org/abs/1907.09386>`_ for technical details of the QWC and
    fully-commuting measurement-partitioning approaches respectively.

    Args:
        observables (list[Observable]): a list of Pauli words (Pauli operation instances and Tensor
            instances thereof)
        coefficients (list[float]): a list of float coefficients, for instance the weights of
            the Pauli words comprising a Hamiltonian
        grouping (str): the binary symmetric relation to use for operator partitioning
        colouring_method (str): the graph-colouring heuristic to use in obtaining the operator
            partitions

    Returns:
        tuple:

            * list[callable]: a list of the post-rotation templates, one
              for each partition
            * list[list[Observable]]: A list of the obtained groupings. Each
              grouping is itself a list of Pauli words diagonal in the
              measurement basis.
            * list[list[float]]: A list of coefficient groupings. Each
              coefficient grouping is itself a list of the partitions
              corresponding coefficients.  Only output if coefficients are
              specified.

    **Example**

    >>> obs = [qml.Y(0), qml.X(0) @ qml.X(1), qml.Z(1)]
    >>> coeffs = [1.43, 4.21, 0.97]
    >>> rotations, groupings, grouped_coeffs = optimize_measurements(obs, coeffs, 'qwc', 'rlf')
    >>> print(rotations)
    [[RY(-1.5707963267948966, wires=[0]), RY(-1.5707963267948966, wires=[1])],
     [RX(1.5707963267948966, wires=[0])]]
    >>> print(groupings)
    [[Z(0) @ Z(1)], [Z(0), Z(1)]]
    >>> print(grouped_coeffs)
    [[4.21], [1.43, 0.97]]
    