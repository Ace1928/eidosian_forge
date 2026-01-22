import contextlib
import dataclasses
import datetime
import time
import uuid
from typing import Any, Dict, Optional, List, TYPE_CHECKING
import cirq
import numpy as np
from cirq import _compat
from cirq.protocols import dataclass_json_dict
from cirq_google.workflow.io import _FilesystemSaver
from cirq_google.workflow.progress import _PrintLogger
from cirq_google.workflow.quantum_executable import (
from cirq_google.workflow.qubit_placement import QubitPlacer, NaiveQubitPlacer
def _try_tuple(k: Any) -> Any:
    """If we serialize a dictionary that had tuple keys, they get turned to json lists."""
    if isinstance(k, list):
        return tuple(k)
    return k