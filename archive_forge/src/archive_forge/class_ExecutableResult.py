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
@dataclasses.dataclass
class ExecutableResult:
    """Results for a `cg.QuantumExecutable`.

    Args:
        spec: The `cg.ExecutableSpec` typifying the `cg.QuantumExecutable`.
        runtime_info: A `cg.RuntimeInfo` dataclass containing information gathered during
            execution of the `cg.QuantumExecutable`.
        raw_data: The `cirq.Result` containing the data from the run.
    """
    spec: Optional[ExecutableSpec]
    runtime_info: RuntimeInfo
    raw_data: cirq.Result

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')