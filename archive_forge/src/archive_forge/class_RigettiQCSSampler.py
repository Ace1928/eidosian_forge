from typing import Optional, Sequence
from pyquil import get_qc
from pyquil.api import QuantumComputer
import cirq
from cirq_rigetti import circuit_transformers as transformers
from cirq_rigetti import circuit_sweep_executors as executors
class RigettiQCSSampler(cirq.Sampler):
    """This class supports running circuits on QCS quantum hardware as well as pyQuil's
    quantum virtual machine (QVM). It implements the `cirq.Sampler` interface and
    thereby supports sampling parameterized circuits across parameter sweeps.
    """

    def __init__(self, quantum_computer: QuantumComputer, executor: executors.CircuitSweepExecutor=_default_executor, transformer: transformers.CircuitTransformer=transformers.default):
        """Initializes a `RigettiQCSSampler`.

        Args:
            quantum_computer: A `pyquil.api.QuantumComputer` against which to run the
                `cirq.Circuit`s.
            executor: A callable that first uses the below `transformer` on `cirq.Circuit` s and
                then executes the transformed circuit on the `quantum_computer`. You may pass your
                own callable or any static method on `CircuitSweepExecutors`.
            transformer: A callable that transforms the `cirq.Circuit` into a `pyquil.Program`.
                You may pass your own callable or any static method on `CircuitTransformers`.
        """
        self._quantum_computer = quantum_computer
        self.executor = executor
        self.transformer = transformer

    def run_sweep(self, program: cirq.AbstractCircuit, params: cirq.Sweepable, repetitions: int=1) -> Sequence[cirq.Result]:
        """This will evaluate results on the circuit for every set of parameters in `params`.

        Args:
            program: Circuit to evaluate for each set of parameters in `params`.
            params: `cirq.Sweepable` of parameters which this function passes to
                `cirq.protocols.resolve_parameters` for evaluating the circuit.
            repetitions: Number of times to run each iteration through the `params`. For a given
                set of parameters, the `cirq.Result` will include a measurement for each repetition.

        Returns:
            A list of `cirq.Result` s.
        """
        resolvers = [r for r in cirq.to_resolvers(params)]
        return self.executor(quantum_computer=self._quantum_computer, circuit=program.unfreeze(copy=False), resolvers=resolvers, repetitions=repetitions, transformer=self.transformer)