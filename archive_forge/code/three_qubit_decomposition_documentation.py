from typing import Union, Tuple, Sequence, List, Optional
import numpy as np
import cirq
from cirq import ops
from cirq import transformers as opt
Calculates the angles for a 4-way multiplexed rotation.

    For example, if we want rz(theta[i]) if the select qubits are in state
    |i>, then, multiplexed_angles returns a[i] that can be used in a circuit
    similar to this:

    ---rz(a[0])-X---rz(a[1])--X--rz(a[2])-X--rz(a[3])--X
                |             |           |            |
    ------------@-------------|-----------@------------|
                              |                        |
    --------------------------@------------------------@

    Args:
        theta: the desired angles for each basis state of the select qubits
    Returns:
        the angles to be used in actual rotations in the circuit implementation
    