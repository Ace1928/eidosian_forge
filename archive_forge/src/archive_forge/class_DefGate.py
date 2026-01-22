import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class DefGate(AbstractInstruction):
    """
    A DEFGATE directive.

    :param name: The name of the newly defined gate.
    :param matrix: The matrix defining this gate.
    :param parameters: list of parameters that are used in this gate
    """

    def __init__(self, name: str, matrix: Union[List[List[Any]], np.ndarray, np.matrix], parameters: Optional[List[Parameter]]=None):
        if not isinstance(name, str):
            raise TypeError('Gate name must be a string')
        if name in RESERVED_WORDS:
            raise ValueError("Cannot use {} for a gate name since it's a reserved word".format(name))
        if isinstance(matrix, list):
            rows = len(matrix)
            if not all([len(row) == rows for row in matrix]):
                raise ValueError('Matrix must be square.')
        elif isinstance(matrix, (np.ndarray, np.matrix)):
            rows, cols = matrix.shape
            if rows != cols:
                raise ValueError('Matrix must be square.')
        else:
            raise TypeError('Matrix argument must be a list or NumPy array/matrix')
        if 0 != rows & rows - 1:
            raise ValueError('Dimension of matrix must be a power of 2, got {0}'.format(rows))
        self.name = name
        self.matrix = np.asarray(matrix)
        if parameters:
            if not isinstance(parameters, list):
                raise TypeError('Paramaters must be a list')
            expressions = [elem for row in self.matrix for elem in row if isinstance(elem, Expression)]
            used_params = {param for exp in expressions for param in _contained_parameters(exp)}
            if set(parameters) != used_params:
                raise ValueError('Parameters list does not match parameters actually used in gate matrix:\nParameters in argument: {}, Parameters in matrix: {}'.format(parameters, used_params))
        else:
            is_unitary = np.allclose(np.eye(rows), self.matrix.dot(self.matrix.T.conj()))
            if not is_unitary:
                raise ValueError('Matrix must be unitary.')
        self.parameters = parameters

    def out(self) -> str:
        """
        Prints a readable Quil string representation of this gate.

        :returns: String representation of a gate
        """

        def format_matrix_element(element: Union[ExpressionDesignator, str]) -> str:
            """
            Formats a parameterized matrix element.

            :param element: The parameterized element to format.
            """
            if isinstance(element, (int, float, complex, np.int_)):
                return format_parameter(element)
            elif isinstance(element, str):
                return element
            elif isinstance(element, Expression):
                return str(element)
            else:
                raise TypeError('Invalid matrix element: %r' % element)
        if self.parameters:
            result = 'DEFGATE {}({}):\n'.format(self.name, ', '.join(map(str, self.parameters)))
        else:
            result = 'DEFGATE {}:\n'.format(self.name)
        for row in self.matrix:
            result += '    '
            fcols = [format_matrix_element(col) for col in row]
            result += ', '.join(fcols)
            result += '\n'
        return result

    def get_constructor(self) -> Union[Callable[..., Gate], Callable[..., Callable[..., Gate]]]:
        """
        :returns: A function that constructs this gate on variable qubit indices. E.g.
                  `mygate.get_constructor()(1) applies the gate to qubit 1.`
        """
        if self.parameters:
            return lambda *params: lambda *qubits: Gate(name=self.name, params=list(params), qubits=list(map(unpack_qubit, qubits)))
        else:
            return lambda *qubits: Gate(name=self.name, params=[], qubits=list(map(unpack_qubit, qubits)))

    def num_args(self) -> int:
        """
        :return: The number of qubit arguments the gate takes.
        """
        rows = len(self.matrix)
        return int(np.log2(rows))