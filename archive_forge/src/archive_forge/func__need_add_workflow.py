import copy
import inspect
from typing import Any, Callable, Dict, Iterable, Optional
from triad import extension_method
from triad.collections.function_wrapper import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function
from fugue.constants import FUGUE_ENTRYPOINT
from fugue.exceptions import FugueInterfacelessError
from fugue.workflow.workflow import FugueWorkflow, WorkflowDataFrame, WorkflowDataFrames
def _need_add_workflow(self, *args: Any, **kwargs: Any):
    if not self._first_annotation_is_workflow:
        return False
    if self._params.get_key_by_index(0) in kwargs:
        return False
    if len(args) > 0 and isinstance(args[0], FugueWorkflow):
        return False
    return True