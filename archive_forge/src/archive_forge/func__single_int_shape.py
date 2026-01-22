import functools
import warnings
from typing import Sequence, Tuple, Optional, Union
import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires
from .measurements import MeasurementShapeError, Sample, SampleMeasurement
from .mid_measure import MeasurementValue
def _single_int_shape(shot_val, num_wires):
    inner_shape = []
    if shot_val != 1:
        inner_shape.append(shot_val)
    if num_wires != 1:
        inner_shape.append(num_wires)
    return tuple(inner_shape)