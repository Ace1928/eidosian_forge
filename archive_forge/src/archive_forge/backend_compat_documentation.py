from __future__ import annotations
import logging
import warnings
from typing import List, Iterable, Any, Dict, Optional
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.backend import QubitProperties
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.providers.models.pulsedefaults import PulseDefaults
from qiskit.providers.options import Options
from qiskit.providers.exceptions import BackendPropertyError
A :class:`qiskit.transpiler.Target` object for the backend.

        :rtype: Target
        