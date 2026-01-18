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
Execute a `cg.QuantumExecutableGroup` according to a `cg.QuantumRuntimeConfiguration`.

    The ExecutableGroupResult's constituent parts will be saved to disk as they become
    available. Within the "{base_data_dir}/{run_id}" directory we save:
        - The `cg.QuantumRuntimeConfiguration` at the start of the execution as a record
          of *how* the executable group was run.
        - A `cg.SharedRuntimeInfo` which is updated throughout the run.
        - An `cg.ExecutableResult` for each `cg.QuantumExecutable` as they become available.
        - A `cg.ExecutableGroupResultFilesystemRecord` which is updated throughout the run.

    Args:
        rt_config: The `cg.QuantumRuntimeConfiguration` specifying how to execute
            `executable_group`.
        executable_group: The `cg.QuantumExecutableGroup` containing the executables to execute.
        base_data_dir: Each data file will be written to the "{base_data_dir}/{run_id}/" directory,
            which must not already exist.

    Returns:
        The `cg.ExecutableGroupResult` containing all data and metadata for an execution.

    Raises:
        NotImplementedError: If an executable uses the `params` field or anything other than
            a BitstringsMeasurement measurement field.
        ValueError: If `base_data_dir` is not a valid directory.
    