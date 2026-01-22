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
class QuantumRuntimeConfiguration:
    """User-requested configuration of how to execute a given `cg.QuantumExecutableGroup`.

    Args:
        processor: The `cg.AbstractEngineProcessor` responsible for running circuits and providing
            device information.
        run_id: A unique `str` identifier for a run. If data already exists for the specified
            `run_id`, an exception will be raised. If not specified, we will generate a UUID4
            run identifier.
        random_seed: An initial seed to make the run deterministic. Otherwise, the default numpy
            seed will be used.
        qubit_placer: A `cg.QubitPlacer` implementation to map executable qubits to device qubits.
            The placer is only called if a given `cg.QuantumExecutable` has a `problem_topology`.
            This subroutine's runtime is keyed by "placement" in `RuntimeInfo.timings_s`.
        target_gateset: If not `None`, compile all circuits to this target gateset prior to
            execution with `cirq.optimize_for_target_gateset`.
    """
    processor_record: 'cg.ProcessorRecord'
    run_id: Optional[str] = None
    random_seed: Optional[int] = None
    qubit_placer: QubitPlacer = NaiveQubitPlacer()
    target_gateset: Optional[cirq.CompilationTargetGateset] = None

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')