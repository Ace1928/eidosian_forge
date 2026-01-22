import itertools
import warnings
from copy import copy
from functools import reduce, wraps
from itertools import combinations
from typing import List, Tuple, Union
from scipy.sparse import kron as sparse_kron
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, convert_to_opmath
from pennylane.ops.op_math.pow import Pow
from pennylane.ops.op_math.sprod import SProd
from pennylane.ops.op_math.sum import Sum
from pennylane.ops.qubit import Hamiltonian
from pennylane.ops.qubit.non_parametric_ops import PauliX, PauliY, PauliZ
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .composite import CompositeOp
class _ProductFactorsGrouping:
    """Utils class used for grouping identical product factors."""
    _identity_map = {'Identity': (1.0, 'Identity'), 'PauliX': (1.0, 'PauliX'), 'PauliY': (1.0, 'PauliY'), 'PauliZ': (1.0, 'PauliZ')}
    _x_map = {'Identity': (1.0, 'PauliX'), 'PauliX': (1.0, 'Identity'), 'PauliY': (1j, 'PauliZ'), 'PauliZ': (-1j, 'PauliY')}
    _y_map = {'Identity': (1.0, 'PauliY'), 'PauliX': (-1j, 'PauliZ'), 'PauliY': (1.0, 'Identity'), 'PauliZ': (1j, 'PauliX')}
    _z_map = {'Identity': (1.0, 'PauliZ'), 'PauliX': (1j, 'PauliY'), 'PauliY': (-1j, 'PauliX'), 'PauliZ': (1.0, 'Identity')}
    _pauli_mult = {'Identity': _identity_map, 'PauliX': _x_map, 'PauliY': _y_map, 'PauliZ': _z_map}
    _paulis = {'PauliX': PauliX, 'PauliY': PauliY, 'PauliZ': PauliZ}

    def __init__(self):
        self._pauli_factors = {}
        self._non_pauli_factors = {}
        self._factors = []
        self.global_phase = 1

    def add(self, factor: Operator):
        """Add factor.

        Args:
            factor (Operator): Factor to add.
        """
        wires = factor.wires
        if isinstance(factor, Prod):
            for prod_factor in factor:
                self.add(prod_factor)
        elif isinstance(factor, Sum):
            self._remove_pauli_factors(wires=wires)
            self._remove_non_pauli_factors(wires=wires)
            self._factors += (factor.operands,)
        elif not isinstance(factor, qml.Identity):
            if isinstance(factor, SProd):
                self.global_phase *= factor.scalar
                factor = factor.base
            if isinstance(factor, (qml.Identity, qml.X, qml.Y, qml.Z)):
                self._add_pauli_factor(factor=factor, wires=wires)
                self._remove_non_pauli_factors(wires=wires)
            else:
                self._add_non_pauli_factor(factor=factor, wires=wires)
                self._remove_pauli_factors(wires=wires)

    def _add_pauli_factor(self, factor: Operator, wires: List[int]):
        """Adds the given Pauli operator to the temporary ``self._pauli_factors`` dictionary. If
        there was another Pauli operator acting on the same wire, the two operators are grouped
        together using the ``self._pauli_mult`` dictionary.

        Args:
            factor (Operator): Factor to be added.
            wires (List[int]): Factor wires. This argument is added to avoid calling
                ``factor.wires`` several times.
        """
        wire = wires[0]
        op2_name = factor.name
        old_coeff, old_word = self._pauli_factors.get(wire, (1, 'Identity'))
        coeff, new_word = self._pauli_mult[old_word][op2_name]
        self._pauli_factors[wire] = (old_coeff * coeff, new_word)

    def _add_non_pauli_factor(self, factor: Operator, wires: List[int]):
        """Adds the given non-Pauli factor to the temporary ``self._non_pauli_factors`` dictionary.
        If there alerady exists an identical operator in the dictionary, the two are grouped
        together.

        If there isn't an identical operator in the dictionary, all non Pauli factors that act on
        the same wires are removed and added to the ``self._factors`` tuple.

        Args:
            factor (Operator): Factor to be added.
            wires (List[int]): Factor wires. This argument is added to avoid calling
                ``factor.wires`` several times.
        """
        if isinstance(factor, Pow):
            exponent = factor.z
            factor = factor.base
        else:
            exponent = 1
        op_hash = factor.hash
        old_hash, old_exponent, old_op = self._non_pauli_factors.get(wires, [None, None, None])
        if isinstance(old_op, (qml.RX, qml.RY, qml.RZ)) and factor.name == old_op.name:
            self._non_pauli_factors[wires] = [op_hash, old_exponent, factor.__class__(factor.data[0] + old_op.data[0], wires).simplify()]
        elif op_hash == old_hash:
            self._non_pauli_factors[wires][1] += exponent
        else:
            self._remove_non_pauli_factors(wires=wires)
            self._non_pauli_factors[wires] = [op_hash, copy(exponent), factor]

    def _remove_non_pauli_factors(self, wires: List[int]):
        """Remove all factors from the ``self._non_pauli_factors`` dictionary that act on the given
        wires and add them to the ``self._factors`` tuple.

        Args:
            wires (List[int]): Wires of the operators to be removed.
        """
        if not self._non_pauli_factors:
            return
        for wire in wires:
            for key, (_, exponent, op) in list(self._non_pauli_factors.items()):
                if wire in key:
                    self._non_pauli_factors.pop(key)
                    if exponent == 0:
                        continue
                    if exponent != 1:
                        op = Pow(base=op, z=exponent).simplify()
                    if not isinstance(op, qml.Identity):
                        self._factors += ((op,),)

    def _remove_pauli_factors(self, wires: List[int]):
        """Remove all Pauli factors from the ``self._pauli_factors`` dictionary that act on the
        given wires and add them to the ``self._factors`` tuple.

        Args:
            wires (List[int]): Wires of the operators to be removed.
        """
        if not self._pauli_factors:
            return
        for wire in wires:
            pauli_coeff, pauli_word = self._pauli_factors.pop(wire, (1, 'Identity'))
            if pauli_word != 'Identity':
                pauli_op = self._paulis[pauli_word](wire)
                self._factors += ((pauli_op,),)
                self.global_phase *= pauli_coeff

    def remove_factors(self, wires: List[int]):
        """Remove all factors from the ``self._pauli_factors`` and ``self._non_pauli_factors``
        dictionaries that act on the given wires and add them to the ``self._factors`` tuple.

        Args:
            wires (List[int]): Wires of the operators to be removed.
        """
        self._remove_pauli_factors(wires=wires)
        self._remove_non_pauli_factors(wires=wires)

    @property
    def factors(self):
        """Grouped factors tuple.

        Returns:
            tuple: Tuple of grouped factors.
        """
        return tuple(self._factors)