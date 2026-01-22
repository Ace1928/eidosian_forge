from __future__ import annotations
from numbers import Real
from collections.abc import Mapping
from typing import Tuple, Union
import numpy as np
from qiskit import QuantumCircuit
from .bindings_array import BindingsArray, BindingsArrayLike
from .observables_array import ObservablesArray, ObservablesArrayLike
from .shape import ShapedMixin
Validate the pub.