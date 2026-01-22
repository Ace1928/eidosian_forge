from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
@property
def _num_thought_containers(self) -> int:
    """The number of 'thought containers' we're currently showing: the
        number of completed thought containers, the history container (if it exists),
        and the current thought container (if it exists).
        """
    count = len(self._completed_thoughts)
    if self._history_container is not None:
        count += 1
    if self._current_thought is not None:
        count += 1
    return count