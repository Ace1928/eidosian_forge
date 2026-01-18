import abc
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING, List, Set, Type
from cirq import _compat, ops, devices
from cirq.devices import noise_utils
Returns the portion of Pauli error from depolarization.