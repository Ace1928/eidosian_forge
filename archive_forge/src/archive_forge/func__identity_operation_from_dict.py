import datetime
import functools
from typing import Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING
from cirq.protocols.json_serialization import ObjectFactory
def _identity_operation_from_dict(qubits, **kwargs):
    return cirq.identity_each(*qubits)