from typing import Sequence, Optional
import pennylane as qml
from pennylane.wires import Wires
from .measurements import StateMeasurement, Purity
Measurement process that computes the purity of the system prior to measurement.

    Please refer to :func:`purity` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
        id (str): custom label given to a measurement instance, can be useful for some
            applications where the instance has to be identified
    