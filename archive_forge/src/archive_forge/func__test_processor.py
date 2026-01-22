import pytest
import numpy as np
import google.protobuf.text_format as text_format
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory
def _test_processor(processor: cg.engine.abstract_processor.AbstractProcessor):
    """Tests an engine instance with some standard commands.
    Also tests the non-Sycamore qubits and gates fail."""
    good_qubit = cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(cirq.X(good_qubit), cirq.measure(good_qubit))
    results = processor.run(circuit, repetitions=100)
    assert np.all(results.measurements[str(good_qubit)] == 1)
    with pytest.raises(RuntimeError, match='requested total repetitions'):
        _ = processor.run(circuit, repetitions=100000000)
    bad_qubit = cirq.GridQubit(10, 10)
    circuit = cirq.Circuit(cirq.X(bad_qubit), cirq.measure(bad_qubit))
    with pytest.raises(ValueError, match='Qubit not on device'):
        _ = processor.run(circuit, repetitions=100)
    circuit = cirq.Circuit(cirq.H(good_qubit), cirq.measure(good_qubit))
    with pytest.raises(ValueError, match='Cannot serialize op'):
        _ = processor.run(circuit, repetitions=100)