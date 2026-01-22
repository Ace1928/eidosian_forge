import pytest
import cirq
class NonMeasurementGate(cirq.Gate):

    def _is_measurement_(self):
        return False

    def _decompose_(self, qubits):
        assert False

    def _measurement_key_name_(self):
        assert False

    def _measurement_key_names_(self):
        assert False

    def _measurement_key_obj_(self):
        assert False

    def _measurement_key_objs_(self):
        assert False

    def num_qubits(self) -> int:
        return 2