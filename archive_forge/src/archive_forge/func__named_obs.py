from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
def _named_obs(self, observable, wires_map: dict):
    """Serializes a Named observable"""
    wires = [wires_map[w] for w in observable.wires]
    if observable.name == 'Identity':
        wires = wires[:1]
    return self.named_obs(observable.name, wires)