import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def _log_text_metrics(self, metrics: Sequence[dict], step: int) -> None:
    if not metrics:
        return
    metrics_summary = _summarize_metrics_for_generated_outputs(metrics)
    for key, value in metrics_summary.items():
        self.experiment.log_metrics(value, prefix=key, step=step)