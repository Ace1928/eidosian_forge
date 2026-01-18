import abc
from typing import Iterable, Sequence, TYPE_CHECKING, List
from cirq import _import, ops, protocols, devices
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
Returns True if an operation is virtual.

        Device-specific subclasses should implement this method to mark any
        operations which their device handles outside the quantum hardware.

        Args:
            op: an operation to check for virtual indicators.

        Returns:
            True if `op` is virtual.
        