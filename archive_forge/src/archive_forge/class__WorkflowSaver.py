import abc
import dataclasses
import os
from typing import Any, Dict, List, TYPE_CHECKING
import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
class _WorkflowSaver(abc.ABC):

    def initialize(self, rt_config: 'cg.QuantumRuntimeConfiguration', shared_rt_info: 'cg.SharedRuntimeInfo'):
        """Initialize a data saving for a workflow execution.

        Args:
            rt_config: The immutable `cg.QuantumRuntimeConfiguation` for this run. This should
                be saved once, likely during initialization.
            shared_rt_info: The current `cg.SharedRuntimeInfo` for saving.
        """

    def consume_result(self, exe_result: 'cg.ExecutableResult', shared_rt_info: 'cg.SharedRuntimeInfo'):
        """Consume an `cg.ExecutableResult` that has been completed.

        Args:
            exe_result: The completed `cg.ExecutableResult` to be saved.
            shared_rt_info: The current `cg.SharedRuntimeInfo` to be saved or updated.
        """

    def finalize(self, shared_rt_info: 'cg.SharedRuntimeInfo'):
        """Called at the end of a workflow execution to finalize data saving.

        Args:
            shared_rt_info: The final `cg.SharedRuntimeInfo` to be saved or updated.
        """