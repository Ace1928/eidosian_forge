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
@_ModuleFunctionWrapper.annotated_param(FugueWorkflow, 'w', matcher=lambda x: inspect.isclass(x) and issubclass(x, FugueWorkflow))
class _FugueWorkflowParam(AnnotatedParam):
    pass