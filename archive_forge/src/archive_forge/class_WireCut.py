from copy import copy
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires  # pylint: disable=unused-import
class WireCut(Operation):
    """WireCut(wires)
    The wire cut operation, used to manually mark locations for wire cuts.

    .. note::

        This operation is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    **Details:**

    * Number of wires: AnyWires
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wires the operation acts on
    """
    num_params = 0
    num_wires = AnyWires
    grad_method = None

    def __init__(self, *params, wires=None, id=None):
        if wires == []:
            raise ValueError(f'{self.__class__.__name__}: wrong number of wires. At least one wire has to be given.')
        super().__init__(*params, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(wires):
        """Representation of the operator as a product of other operators (static method).

        Since this operator is a placeholder inside a circuit, it decomposes into an empty list.

        Args:
            wires (Any, Wires): Wire that the operator acts on.

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> print(qml.WireCut.compute_decomposition(0))
        []

        """
        return []

    def label(self, decimals=None, base_label=None, cache=None):
        return '//'

    def adjoint(self):
        return WireCut(wires=self.wires)

    def pow(self, z):
        return [copy(self)]