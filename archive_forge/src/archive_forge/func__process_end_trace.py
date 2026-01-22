from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict
from langchain_core.tracers import BaseTracer
def _process_end_trace(self, run: 'Run') -> None:
    if not run.parent_run_id:
        pass
    else:
        span = self._span_map[run.id]
        span.set_outputs(outputs=run.outputs)
        span.__api__end__()