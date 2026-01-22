from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import copy
from typing import Generic, TypeVar
from qiskit.circuit import QuantumCircuit
from qiskit.providers import JobV1 as Job
from ..containers.primitive_result import PrimitiveResult
from ..containers.pub_result import PubResult
from ..containers.sampler_pub import SamplerPubLike
from . import validation
from .base_primitive import BasePrimitive
from .base_primitive_job import BasePrimitiveJob
class BaseSamplerV2(ABC):
    """Sampler V2 base class.

    A Sampler returns samples of quantum circuit outputs.

    All sampler implementations must implement default value for the ``shots`` in the
    :meth:`.run` method if ``None`` is given both as a ``kwarg`` and in all of the pubs.
    """

    @abstractmethod
    def run(self, pubs: Iterable[SamplerPubLike], *, shots: int | None=None) -> BasePrimitiveJob[PrimitiveResult[PubResult]]:
        """Run and collect samples from each pub.

        Args:
            pubs: An iterable of pub-like objects. For example, a list of circuits
                  or tuples ``(circuit, parameter_values)``.
            shots: The total number of shots to sample for each sampler pub that does
                   not specify its own shots. If ``None``, the primitive's default
                   shots value will be used, which can vary by implementation.

        Returns:
            The job object of Sampler's result.
        """