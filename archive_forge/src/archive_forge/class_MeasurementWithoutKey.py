import pytest
import cirq
class MeasurementWithoutKey:

    def _is_measurement_(self):
        return True