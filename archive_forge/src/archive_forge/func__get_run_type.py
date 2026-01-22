from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict
from langchain_core.tracers import BaseTracer
def _get_run_type(run: 'Run') -> str:
    if isinstance(run.run_type, str):
        return run.run_type
    elif hasattr(run.run_type, 'value'):
        return run.run_type.value
    else:
        return str(run.run_type)