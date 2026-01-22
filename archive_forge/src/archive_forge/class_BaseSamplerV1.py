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
class BaseSamplerV1(BasePrimitive, Generic[T]):
    """Sampler V1 base class

    Base class of Sampler that calculates quasi-probabilities of bitstrings from quantum circuits.

    A sampler is initialized with an empty parameter set. The sampler is used to
    create a :class:`~qiskit.providers.JobV1`, via the :meth:`qiskit.primitives.Sampler.run()`
    method. This method is called with the following parameters

    * quantum circuits (:math:`\\psi_i(\\theta)`): list of (parameterized) quantum circuits.
      (a list of :class:`~qiskit.circuit.QuantumCircuit` objects)

    * parameter values (:math:`\\theta_k`): list of sets of parameter values
      to be bound to the parameters of the quantum circuits.
      (list of list of float)

    The method returns a :class:`~qiskit.providers.JobV1` object, calling
    :meth:`qiskit.providers.JobV1.result()` yields a :class:`~qiskit.primitives.SamplerResult`
    object, which contains probabilities or quasi-probabilities of bitstrings,
    plus optional metadata like error bars in the samples.

    Here is an example of how sampler is used.

    .. code-block:: python

        from qiskit.primitives import Sampler
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import RealAmplitudes

        # a Bell circuit
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()

        # two parameterized circuits
        pqc = RealAmplitudes(num_qubits=2, reps=2)
        pqc.measure_all()
        pqc2 = RealAmplitudes(num_qubits=2, reps=3)
        pqc2.measure_all()

        theta1 = [0, 1, 1, 2, 3, 5]
        theta2 = [0, 1, 2, 3, 4, 5, 6, 7]

        # initialization of the sampler
        sampler = Sampler()

        # Sampler runs a job on the Bell circuit
        job = sampler.run(circuits=[bell], parameter_values=[[]], parameters=[[]])
        job_result = job.result()
        print([q.binary_probabilities() for q in job_result.quasi_dists])

        # Sampler runs a job on the parameterized circuits
        job2 = sampler.run(
            circuits=[pqc, pqc2],
            parameter_values=[theta1, theta2],
            parameters=[pqc.parameters, pqc2.parameters])
        job_result = job2.result()
        print([q.binary_probabilities() for q in job_result.quasi_dists])

    """
    __hash__ = None

    def __init__(self, *, options: dict | None=None):
        """
        Args:
            options: Default options.
        """
        super().__init__(options)

    def run(self, circuits: QuantumCircuit | Sequence[QuantumCircuit], parameter_values: Sequence[float] | Sequence[Sequence[float]] | None=None, **run_options) -> T:
        """Run the job of the sampling of bitstrings.

        Args:
            circuits: One of more circuit objects.
            parameter_values: Parameters to be bound to the circuit.
            run_options: Backend runtime options used for circuit execution.

        Returns:
            The job object of the result of the sampler. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``.

        Raises:
            ValueError: Invalid arguments are given.
        """
        circuits, parameter_values = validation._validate_sampler_args(circuits, parameter_values)
        run_opts = copy(self.options)
        run_opts.update_options(**run_options)
        return self._run(circuits, parameter_values, **run_opts.__dict__)

    @abstractmethod
    def _run(self, circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...], **run_options) -> T:
        raise NotImplementedError('The subclass of BaseSampler must implement `_run` method.')