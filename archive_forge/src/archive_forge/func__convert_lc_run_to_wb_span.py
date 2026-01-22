from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def _convert_lc_run_to_wb_span(self, run: Run) -> 'Span':
    """Utility to convert any generic LangChain Run into a W&B Trace Span.
        :param run: The LangChain Run to convert.
        :return: The converted W&B Trace Span.
        """
    if run.run_type == 'llm':
        return self._convert_llm_run_to_wb_span(run)
    elif run.run_type == 'chain':
        return self._convert_chain_run_to_wb_span(run)
    elif run.run_type == 'tool':
        return self._convert_tool_run_to_wb_span(run)
    else:
        return self._convert_run_to_wb_span(run)