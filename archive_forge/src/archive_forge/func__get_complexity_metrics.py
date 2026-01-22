import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def _get_complexity_metrics(self, text: str) -> dict:
    """Compute text complexity metrics using textstat.

        Parameters:
            text (str): The text to analyze.

        Returns:
            (dict): A dictionary containing the complexity metrics.
        """
    resp = {}
    if self.complexity_metrics:
        text_complexity_metrics = _fetch_text_complexity_metrics(text)
        resp.update(text_complexity_metrics)
    return resp