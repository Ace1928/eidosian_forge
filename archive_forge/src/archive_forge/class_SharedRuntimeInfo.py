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
class SharedRuntimeInfo:
    """Runtime information common to all `cg.QuantumExecutable`s in an execution of a
    `cg.QuantumExecutableGroup`.

    There is one `cg.SharedRuntimeInfo` per `cg.ExecutableGroupResult`.

    Args:
        run_id: A unique `str` identifier for this run.
        device: The actual device used during execution, not just its processor_id
    """
    run_id: str
    device: Optional[cirq.Device] = None
    run_start_time: Optional[datetime.datetime] = None
    run_end_time: Optional[datetime.datetime] = None

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')