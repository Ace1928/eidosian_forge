from functools import lru_cache, reduce
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops.qubit.parametric_ops_multi_qubit import PauliRot
class TmpPauliRot(PauliRot):
    """A custom version of ``PauliRot`` that is inserted with rotation angle zero when
    decomposing ``SpecialUnitary``. The differentiation logic makes use of the gradient
    recipe of ``PauliRot``, but deactivates the matrix property so that a decomposition
    of differentiated tapes is forced. During this decomposition, this private operation
    removes itself if its angle remained at zero.

    For details see :class:`~.PauliRot`.

    .. warning::

        Do not add this operation to the supported operations of any device.
        Wrong results and/or severe performance degradations may result.
    """
    has_matrix = False

    @staticmethod
    def compute_decomposition(theta, wires, pauli_word):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.TmpPauliRot.decomposition`.

        Args:
            theta (float): rotation angle :math:`\\theta`
            wires (Iterable, Wires): the wires the operation acts on
            pauli_word (string): the Pauli word defining the rotation

        Returns:
            list[Operator]: decomposition into an empty list of operations for
            vanishing ``theta``, or into a list containing a single :class:`~.PauliRot`
            for non-zero ``theta``.

        .. note::

            This operation is used in a differentiation pipeline of :class:`~.SpecialUnitary`
            and most likely should not be created manually by users.
        """
        if qml.math.isclose(theta, theta * 0) and (not qml.math.requires_grad(theta)):
            return []
        return [PauliRot(theta, pauli_word, wires)]

    def __repr__(self):
        return f'TmpPauliRot({self.data[0]}, {self.hyperparameters['pauli_word']}, wires={self.wires.tolist()})'