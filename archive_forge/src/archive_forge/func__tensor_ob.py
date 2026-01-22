from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
def _tensor_ob(self, observable, wires_map: dict):
    """Serialize a tensor observable"""
    assert isinstance(observable, Tensor)
    return self.tensor_obs([self._ob(obs, wires_map) for obs in observable.obs])