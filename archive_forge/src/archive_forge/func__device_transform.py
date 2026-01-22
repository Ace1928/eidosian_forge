import functools
import os
import copy
import warnings
import types
from typing import Sequence
import pennylane as qml
from pennylane.typing import ResultBatch
def _device_transform(self, original_device, targs, tkwargs):
    """Apply the transform on a device"""
    if self._expand_transform:
        raise TransformError('Device transform does not support expand transforms.')
    if self._is_informative:
        raise TransformError('Device transform does not support informative transforms.')
    if self._final_transform:
        raise TransformError('Device transform does not support final transforms.')

    class TransformedDevice(type(original_device)):
        """A transformed device with updated preprocess method."""

        def __init__(self, original_device, transform):
            for key, value in original_device.__dict__.items():
                self.__setattr__(key, value)
            self.transform = transform
            self._original_device = original_device

        def __repr__(self):
            return f'Transformed Device({original_device.__repr__()} with additional preprocess transform {self.transform})'

        def preprocess(self, execution_config: qml.devices.ExecutionConfig=qml.devices.DefaultExecutionConfig):
            """This function updates the original device transform program to be applied."""
            program, config = self.original_device.preprocess(execution_config)
            program.push_back(TransformContainer(self.transform, targs, tkwargs))
            return (program, config)

        @property
        def original_device(self):
            """Return the original device."""
            return self._original_device
    return TransformedDevice(original_device, self._transform)