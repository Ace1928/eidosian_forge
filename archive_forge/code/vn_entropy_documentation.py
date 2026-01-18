from typing import Sequence, Optional
import pennylane as qml
from pennylane.wires import Wires
from .measurements import StateMeasurement, VnEntropy
int: returns an integer hash uniquely representing the measurement process