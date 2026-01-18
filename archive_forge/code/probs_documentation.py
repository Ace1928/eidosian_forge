from typing import Sequence, Tuple
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from .measurements import Probability, SampleMeasurement, StateMeasurement
from .mid_measure import MeasurementValue
Count the occurrences of sampled indices and convert them to relative
        counts in order to estimate their occurrence probability.