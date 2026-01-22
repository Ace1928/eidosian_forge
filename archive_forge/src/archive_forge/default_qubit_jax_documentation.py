import numpy as np
import pennylane as qml
from pennylane.devices import DefaultQubitLegacy
from pennylane.pulse import ParametrizedEvolution
from pennylane.typing import TensorLike
dy/dt = -i H(t) y