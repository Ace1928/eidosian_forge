from typing import Any, Callable, List, Optional, Type, Union
import pandas as pd
from triad import ParamDict, assert_or_throw
from triad.utils.convert import to_instance
from .._utils.registry import fugue_plugin
from ..exceptions import FuguePluginsRegistrationError
from .execution_engine import (
from .native_execution_engine import NativeExecutionEngine
def try_get_context_execution_engine() -> Optional[ExecutionEngine]:
    """If the global execution engine is set (see
    :meth:`~fugue.execution.execution_engine.ExecutionEngine.set_global`)
    or the context is set (see
    :meth:`~fugue.execution.execution_engine.ExecutionEngine.as_context`),
    then return the engine, else return None
    """
    engine = _FUGUE_EXECUTION_ENGINE_CONTEXT.get()
    if engine is None:
        engine = _FUGUE_GLOBAL_EXECUTION_ENGINE_CONTEXT.get()
    return engine