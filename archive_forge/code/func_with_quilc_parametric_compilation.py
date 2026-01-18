from typing import Any, cast, Dict, Optional, Sequence, Union
from pyquil import Program
from pyquil.api import QuantumComputer, QuantumExecutable
from pyquil.quilbase import Declare
import cirq
import sympy
from typing_extensions import Protocol
from cirq_rigetti.logging import logger
from cirq_rigetti import circuit_transformers as transformers
def with_quilc_parametric_compilation(*, quantum_computer: QuantumComputer, circuit: cirq.Circuit, resolvers: Sequence[cirq.ParamResolverOrSimilarType], repetitions: int, transformer: transformers.CircuitTransformer=transformers.default) -> Sequence[cirq.Result]:
    """This `CircuitSweepExecutor` will compile the `circuit` using quilc as a
    parameterized `pyquil.api.QuantumExecutable` and on each iteration of
    `resolvers`, rather than resolving the `circuit` with `cirq.protocols.resolve_parameters`,
    it will attempt to cast the resolver to a dict and pass it as a memory map to
    to `pyquil.api.QuantumComputer`.

    Args:
        quantum_computer: The `pyquil.api.QuantumComputer` against which to execute the circuit.
        circuit: The `cirq.Circuit` to transform into a `pyquil.Program` and executed on the
            `quantum_computer`.
        resolvers: A sequence of parameter resolvers that this executor will write to memory
            on a copy of the `pyquil.api.QuantumExecutable` for each parameter sweep.
        repetitions: Number of times to run each iteration through the `resolvers`. For a given
            resolver, the `cirq.Result` will include a measurement for each repetition.
        transformer: A callable that transforms the `cirq.Circuit` into a `pyquil.Program`.
            You may pass your own callable or any function from `cirq_rigetti.circuit_transformers`.

    Returns:
        A list of `cirq.Result`, each corresponding to a resolver in `resolvers`.
    """
    program, measurement_id_map = transformer(circuit=circuit)
    program = _prepend_real_declarations(program=program, resolvers=resolvers)
    program.wrap_in_numshots_loop(repetitions)
    executable = quantum_computer.compile(program)
    cirq_results = []
    for resolver in resolvers:
        memory_map = _get_param_dict(resolver)
        logger.debug(f'running pre-compiled parametric circuit with parameters {memory_map}')
        result = _execute_and_read_result(quantum_computer, executable.copy(), measurement_id_map, resolver, memory_map=memory_map)
        cirq_results.append(result)
    return cirq_results