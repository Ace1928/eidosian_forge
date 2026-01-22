import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
class MyMeasurement(MeasurementTransform):
    """Dummy measurement transform."""

    def process(self, tape, device):
        return 1