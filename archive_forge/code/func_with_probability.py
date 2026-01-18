import abc
import functools
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, value
from cirq._import import LazyLoader
from cirq._compat import __cirq_debug__, cached_method
from cirq.type_workarounds import NotImplementedType
from cirq.ops import control_values as cv
def with_probability(self, probability: 'cirq.TParamVal') -> 'cirq.Operation':
    """Creates a probabilistic channel with this operation.

        Args:
            probability: floating point value between 0 and 1, giving the
                probability this gate is applied.

        Returns:
            `cirq.RandomGateChannel` that applies `self` with probability
                `probability` and the identity with probability `1-p`.

        Raises:
            NotImplementedError: if called on an operation that lacks a gate.
        """
    gate = self.gate
    if gate is None:
        raise NotImplementedError('with_probability on gateless operation.')
    if probability == 1:
        return self
    return ops.random_gate_channel.RandomGateChannel(sub_gate=gate, probability=probability).on(*self.qubits)