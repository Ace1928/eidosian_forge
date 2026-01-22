import copy
import functools
from warnings import warn
from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Tuple, Optional, Union
import pennylane as qml
from pennylane.operation import Operator, DecompositionUndefinedError, EigvalsUndefinedError
from pennylane.pytrees import register_pytree
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .shots import Shots
from a classical shadow measurement"""
@staticmethod
@functools.lru_cache()
def _get_num_basis_states(num_wires, device):
    """Auxiliary function to determine the number of basis states given the
        number of systems and a quantum device.

        This function is meant to be used with the Probability measurement to
        determine how many outcomes there will be. With qubit based devices
        we'll have two outcomes for each subsystem. With continuous variable
        devices that impose a Fock cutoff the number of basis states per
        subsystem equals the cutoff value.

        Args:
            num_wires (int): the number of qubits/qumodes
            device (pennylane.Device): a PennyLane device

        Returns:
            int: the number of basis states
        """
    cutoff = getattr(device, 'cutoff', None)
    base = 2 if cutoff is None else cutoff
    return base ** num_wires