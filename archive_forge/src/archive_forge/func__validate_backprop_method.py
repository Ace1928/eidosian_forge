import copy
import functools
import inspect
import warnings
from collections.abc import Sequence
from typing import Union
import logging
import pennylane as qml
from pennylane import Device
from pennylane.measurements import CountsMP, MidMeasureMP, Shots
from pennylane.tape import QuantumTape, QuantumScript
from .execution import INTERFACE_MAP, SUPPORTED_INTERFACES
from .set_shots import set_shots
@staticmethod
def _validate_backprop_method(device, interface, shots=None):
    if shots is not None or _get_device_shots(device):
        raise qml.QuantumFunctionError('Backpropagation is only supported when shots=None.')
    if isinstance(device, qml.devices.Device):
        config = qml.devices.ExecutionConfig(gradient_method='backprop', interface=interface)
        if device.supports_derivatives(config):
            return ('backprop', {}, device)
        raise qml.QuantumFunctionError(f'Device {device.name} does not support backprop with {config}')
    mapped_interface = INTERFACE_MAP.get(interface, interface)
    backprop_interface = device.capabilities().get('passthru_interface', None)
    if backprop_interface is not None:
        if mapped_interface == backprop_interface:
            return ('backprop', {}, device)
        raise qml.QuantumFunctionError(f"Device {device.short_name} only supports diff_method='backprop' when using the {backprop_interface} interface.")
    backprop_devices = device.capabilities().get('passthru_devices', None)
    if backprop_devices is not None:
        if mapped_interface in backprop_devices:
            if backprop_devices[mapped_interface] == device.short_name:
                return ('backprop', {}, device)
            expand_fn = device.expand_fn
            batch_transform = device.batch_transform
            device = qml.device(backprop_devices[mapped_interface], wires=device.wires, shots=device.shots)
            device.expand_fn = expand_fn
            device.batch_transform = batch_transform
            return ('backprop', {}, device)
        raise qml.QuantumFunctionError(f"Device {device.short_name} only supports diff_method='backprop' when using the {list(backprop_devices.keys())} interfaces.")
    raise qml.QuantumFunctionError(f'The {device.short_name} device does not support native computations with autodifferentiation frameworks.')