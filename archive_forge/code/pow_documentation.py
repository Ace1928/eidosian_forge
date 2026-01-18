import copy
from typing import Union
from scipy.linalg import fractional_matrix_power
import pennylane as qml
from pennylane import math as qmlmath
from pennylane.operation import (
from pennylane.ops.identity import Identity
from pennylane.queuing import QueuingManager, apply
from .symbolicop import ScalarSymbolicOp
Generator of an operator that is in single-parameter-form.

        The generator of a power operator is ``z`` times the generator of the
        base matrix.

        .. math::

            U(\phi)^z = e^{i\phi (z G)}

        See also :func:`~.generator`
        