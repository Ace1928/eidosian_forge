import numpy as np
import pytest
import sympy
import cirq
class OnlyMeasurementsDevice(cirq.Device):

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        if not cirq.is_measurement(operation):
            raise ValueError(f'{operation} is not a measurement and this device only measures!')