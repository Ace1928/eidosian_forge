from dataclasses import replace
from functools import partial
from typing import Union, Tuple, Sequence
import concurrent.futures
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.measurements import (
from pennylane.ops.qubit.observables import BasisStateProjector
from . import Device
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .default_qubit import accepted_sample_measurement
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
@qml.transform
def _validate_channels(tape, name='device'):
    """Validates the channels for a circuit."""
    if not tape.shots and any((isinstance(op, qml.operation.Channel) for op in tape.operations)):
        raise qml.DeviceError(f'Channel not supported on {name} without finite shots.')
    return ((tape,), null_postprocessing)