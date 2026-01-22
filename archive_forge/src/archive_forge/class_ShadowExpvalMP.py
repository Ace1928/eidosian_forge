import copy
from collections.abc import Iterable
from typing import Optional, Union, Sequence
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires
from .measurements import MeasurementShapeError, MeasurementTransform, Shadow, ShadowExpval
class ShadowExpvalMP(MeasurementTransform):
    """Measures the expectation value of an operator using the classical shadow measurement process.

    Please refer to :func:`shadow_expval` for detailed documentation.

    Args:
        H (Operator, Sequence[Operator]): Operator or list of Operators to compute the expectation value over.
        seed (Union[int, None]): The seed used to generate the random measurements
        k (int): Number of equal parts to split the shadow's measurements to compute the median of means.
            ``k=1`` corresponds to simply taking the mean over all measurements.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    def _flatten(self):
        metadata = (('seed', self.seed), ('k', self.k))
        return ((self.H,), metadata)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], **dict(metadata))

    def __init__(self, H: Union[Operator, Sequence], seed: Optional[int]=None, k: int=1, id: Optional[str]=None):
        self.seed = seed
        self.H = H
        self.k = k
        super().__init__(id=id)

    def process(self, tape, device):
        bits, recipes = qml.classical_shadow(wires=self.wires, seed=self.seed).process(tape, device)
        shadow = qml.shadows.ClassicalShadow(bits, recipes, wire_map=self.wires.tolist())
        return shadow.expval(self.H, self.k)

    def process_state_with_shots(self, state: Sequence[complex], wire_order: Wires, shots: int, rng=None):
        """Process the given quantum state with the given number of shots

        Args:
            state (Sequence[complex]): quantum state
            wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
                dimension :math:`2^n` acts on a subspace of :math:`n` wires
            shots (int): The number of shots
            rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used.

        Returns:
            float: The estimate of the expectation value.
        """
        bits, recipes = qml.classical_shadow(wires=self.wires, seed=self.seed).process_state_with_shots(state, wire_order, shots, rng=rng)
        shadow = qml.shadows.ClassicalShadow(bits, recipes, wire_map=self.wires.tolist())
        return shadow.expval(self.H, self.k)

    @property
    def samples_computational_basis(self):
        return False

    @property
    def numeric_type(self):
        return float

    @property
    def return_type(self):
        return ShadowExpval

    def shape(self, device, shots):
        return ()

    @property
    def wires(self):
        """The wires the measurement process acts on.

        This is the union of all the Wires objects of the measurement.
        """
        if isinstance(self.H, Iterable):
            return Wires.all_wires([h.wires for h in self.H])
        return self.H.wires

    def queue(self, context=qml.QueuingManager):
        """Append the measurement process to an annotated queue, making sure
        the observable is not queued"""
        Hs = self.H if isinstance(self.H, Iterable) else [self.H]
        for H in Hs:
            context.remove(H)
        context.append(self)
        return self

    def __copy__(self):
        H_copy = [copy.copy(H) for H in self.H] if isinstance(self.H, Iterable) else copy.copy(self.H)
        return self.__class__(H=H_copy, k=self.k, seed=self.seed)