import functools
import itertools
from collections import defaultdict
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
import pennylane.math as qnp
from pennylane import (
from pennylane.measurements import CountsMP, MutualInfoMP, SampleMP, StateMP, VnEntropyMP, PurityMP
from pennylane.operation import Channel
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane.wires import Wires
from .._version import __version__
def _apply_diagonal_unitary(self, eigvals, wires):
    """Apply a diagonal unitary gate specified by a list of eigenvalues. This method uses
        the fact that the unitary is diagonal for a more efficient implementation.

        Args:
            eigvals (array): eigenvalues (phases) of the diagonal unitary
            wires (Wires): target wires
        """
    channel_wires = self.map_wires(wires)
    eigvals = qnp.stack(eigvals)
    eigvals = qnp.cast(qnp.reshape(eigvals, [2] * len(channel_wires)), dtype=self.C_DTYPE)
    state_indices = ABC[:2 * self.num_wires]
    row_wires_list = channel_wires.tolist()
    row_indices = ''.join(ABC_ARRAY[row_wires_list].tolist())
    col_wires_list = [w + self.num_wires for w in row_wires_list]
    col_indices = ''.join(ABC_ARRAY[col_wires_list].tolist())
    einsum_indices = f'{row_indices},{state_indices},{col_indices}->{state_indices}'
    self._state = qnp.einsum(einsum_indices, eigvals, self._state, qnp.conj(eigvals))