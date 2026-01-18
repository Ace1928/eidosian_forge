from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
import numbers
import abc
import numpy as np
import cirq
from cirq.circuits import circuit_operation
from cirq_google.api import v2
from cirq_google.serialization.arg_func_langs import arg_to_proto
Returns the cirq.google.api.v2.CircuitOperation message as a proto dict.

        Note that this function requires constants and raw_constants to be
        pre-populated with the circuit in op.
        