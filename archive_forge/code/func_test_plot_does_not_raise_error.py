import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
@pytest.mark.usefixtures('closefigures')
def test_plot_does_not_raise_error():

    class _TimeDependentDecay(cirq.NoiseModel):

        def noisy_moment(self, moment, system_qubits):
            duration = max((op.gate.duration for op in moment.operations if isinstance(op.gate, cirq.WaitGate)), default=cirq.Duration(nanos=1))
            yield cirq.amplitude_damp(1 - 0.99 ** duration.total_nanos()).on_each(system_qubits)
            yield moment
    results = cirq.experiments.t2_decay(sampler=cirq.DensityMatrixSimulator(noise=_TimeDependentDecay()), qubit=cirq.GridQubit(0, 0), num_points=3, repetitions=10, max_delay=cirq.Duration(nanos=500))
    results.plot_expectations()
    results.plot_bloch_vector()