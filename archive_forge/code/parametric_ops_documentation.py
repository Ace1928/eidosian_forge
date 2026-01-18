import functools
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TRZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle
            subspace (Sequence[int]): the 2D subspace on which to apply operation

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.TRZ.compute_matrix(torch.tensor(0.5), subspace=(0, 2))
        tensor([[0.9689-0.2474j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.2474j]])
        