from copy import copy
import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops import Sum
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from pennylane.typing import TensorLike
from pennylane.wires import Wires
def H_parametrized(self, params, t):
    """The parametrized terms of the Hamiltonian for the specified parameters and time.

        Args:
            params(tensor_like): the parameters values used to evaluate the operators
            t(float): the time at which the operator is evaluated

        Returns:
            Operator: a ``Sum`` of ``SProd`` operators (or a single
            ``SProd`` operator in the event that there is only one term in ``H_parametrized``).
        """
    coeffs = [f(param, t) for f, param in zip(self.coeffs_parametrized, params)]
    return sum((qml.s_prod(c, o) for c, o in zip(coeffs, self.ops_parametrized))) if coeffs else 0