import abc
import itertools
import warnings
from collections import defaultdict
from typing import Union, List
import inspect
import logging
import numpy as np
import pennylane as qml
from pennylane import Device, DeviceError
from pennylane.math import multiply as qmlmul
from pennylane.math import sum as qmlsum
from pennylane.measurements import (
from pennylane.resource import Resources
from pennylane.operation import operation_derivative, Operation
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
Returns the gates that diagonalize the measured wires such that they
        are in the eigenbasis of the circuit observables.

        Note that this exists as a method of the Device class to enable child classes to
        override the implementation if necessary (for example, to skip computing rotation
        gates for a measurement that doesn't need them).

        Args:
            circuit (~.tape.QuantumTape): The circuit containing observables that may need diagonalizing

        Returns:
            List[~.Operation]: the operations that diagonalize the observables
        