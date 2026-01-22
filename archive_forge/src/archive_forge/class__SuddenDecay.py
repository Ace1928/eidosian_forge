import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
class _SuddenDecay(cirq.NoiseModel):

    def noisy_moment(self, moment, system_qubits):
        duration = max((op.gate.duration for op in moment.operations if isinstance(op.gate, cirq.WaitGate)), default=cirq.Duration())
        if duration > cirq.Duration(nanos=500):
            yield cirq.amplitude_damp(1).on_each(system_qubits)
        yield moment