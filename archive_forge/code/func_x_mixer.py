import itertools
import functools
from typing import Iterable, Union
import networkx as nx
import rustworkx as rx
import pennylane as qml
from pennylane.wires import Wires
def x_mixer(wires: Union[Iterable, Wires]):
    """Creates a basic Pauli-X mixer Hamiltonian.

    This Hamiltonian is defined as:

    .. math:: H_M \\ = \\ \\displaystyle\\sum_{i} X_{i},

    where :math:`i` ranges over all wires, and :math:`X_i`
    denotes the Pauli-X operator on the :math:`i`-th wire.

    This is mixer is used in *A Quantum Approximate Optimization Algorithm*
    by Edward Farhi, Jeffrey Goldstone, Sam Gutmann [`arXiv:1411.4028 <https://arxiv.org/abs/1411.4028>`__].

    Args:
        wires (Iterable or Wires): The wires on which the Hamiltonian is applied

    Returns:
        Hamiltonian: Mixer Hamiltonian

    **Example**

    The mixer Hamiltonian can be called as follows:

    >>> from pennylane import qaoa
    >>> wires = range(3)
    >>> mixer_h = qaoa.x_mixer(wires)
    >>> print(mixer_h)
      (1) [X0]
    + (1) [X1]
    + (1) [X2]
    """
    wires = Wires(wires)
    coeffs = [1 for w in wires]
    obs = [qml.X(w) for w in wires]
    H = qml.Hamiltonian(coeffs, obs)
    H.grouping_indices = [list(range(len(H.ops)))]
    return H