import functools
import itertools
from string import ascii_letters as ABC
from typing import List
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import BasisState, DeviceError, QubitDevice, StatePrep, Snapshot
from pennylane.devices.qubit import measure
from pennylane.operation import Operation
from pennylane.ops import Sum
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane.pulse import ParametrizedEvolution
from pennylane.measurements import ExpectationMP
from pennylane.typing import TensorLike
from pennylane.wires import WireError
from .._version import __version__

        Returns the measured bits and recipes in the classical shadow protocol.

        The protocol is described in detail in the `classical shadows paper <https://arxiv.org/abs/2002.08953>`_.
        This measurement process returns the randomized Pauli measurements (the ``recipes``)
        that are performed for each qubit and snapshot as an integer:

        - 0 for Pauli X,
        - 1 for Pauli Y, and
        - 2 for Pauli Z.

        It also returns the measurement results (the ``bits``); 0 if the 1 eigenvalue
        is sampled, and 1 if the -1 eigenvalue is sampled.

        The device shots are used to specify the number of snapshots. If ``T`` is the number
        of shots and ``n`` is the number of qubits, then both the measured bits and the
        Pauli measurements have shape ``(T, n)``.

        This implementation leverages vectorization and offers a significant speed-up over
        the generic implementation.

        .. Note::

            This method internally calls ``np.einsum`` which supports at most 52 indices,
            thus the classical shadow measurement for this device supports at most 52
            qubits.

        .. seealso:: :func:`~pennylane.classical_shadow`

        Args:
            obs (~.pennylane.measurements.ClassicalShadowMP): The classical shadow measurement process
            circuit (~.tape.QuantumTape): The quantum tape that is being executed

        Returns:
            tensor_like[int]: A tensor with shape ``(2, T, n)``, where the first row represents
            the measured bits and the second represents the recipes used.
        