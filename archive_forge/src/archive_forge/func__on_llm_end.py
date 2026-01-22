from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict
from langchain_core.tracers import BaseTracer
def _on_llm_end(self, run: 'Run') -> None:
    """Process the LLM Run."""
    self._process_end_trace(run)