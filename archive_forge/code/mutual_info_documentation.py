from copy import copy
from typing import Sequence, Optional
import pennylane as qml
from pennylane.wires import Wires
from .measurements import MutualInfo, StateMeasurement
int: returns an integer hash uniquely representing the measurement process