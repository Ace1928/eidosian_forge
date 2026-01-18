from typing import Any, cast, Dict, Optional, Sequence, Union
from pyquil import Program
from pyquil.api import QuantumComputer, QuantumExecutable
from pyquil.quilbase import Declare
import cirq
import sympy
from typing_extensions import Protocol
from cirq_rigetti.logging import logger
from cirq_rigetti import circuit_transformers as transformers
def without_quilc_compilation(*, quantum_computer: QuantumComputer, circuit: cirq.Circuit, resolvers: Sequence[cirq.ParamResolverOrSimilarType], repetitions: int, transformer: transformers.CircuitTransformer=transformers.default) -> Sequence[cirq.Result]:
    """This `CircuitSweepExecutor` will bypass quilc entirely, treating the transformed
    `cirq.Circuit` as native Quil.

    Args:
        quantum_computer: The `pyquil.api.QuantumComputer` against which to execute the circuit.
        circuit: The `cirq.Circuit` to transform into a `pyquil.Program` and executed on the
            `quantum_computer`.
        resolvers: A sequence of parameter resolvers that `cirq.protocols.resolve_parameters` will
            use to fully resolve the circuit.
        repetitions: Number of times to run each iteration through the `resolvers`. For a given
            resolver, the `cirq.Result` will include a measurement for each repetition.
        transformer: A callable that transforms the `cirq.Circuit` into a `pyquil.Program`.
            You may pass your own callable or any function from `cirq_rigetti.circuit_transformers`.

    Returns:
        A list of `cirq.Result`, each corresponding to a resolver in `resolvers`.
    """
    cirq_results = []
    for resolver in resolvers:
        resolved_circuit = cirq.protocols.resolve_parameters(circuit, resolver)
        program, measurement_id_map = transformer(circuit=resolved_circuit)
        program = program.wrap_in_numshots_loop(repetitions)
        executable = quantum_computer.compile(program, optimize=False, to_native_gates=False)
        result = _execute_and_read_result(quantum_computer, executable, measurement_id_map, resolver)
        cirq_results.append(result)
    return cirq_results