from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict
from langchain_core.tracers import BaseTracer
def _on_llm_start(self, run: 'Run') -> None:
    """Process the LLM Run upon start."""
    self._process_start_trace(run)