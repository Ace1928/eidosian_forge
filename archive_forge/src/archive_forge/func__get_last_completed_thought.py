from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
def _get_last_completed_thought(self) -> Optional[LLMThought]:
    """Return our most recent completed LLMThought, or None if we don't have one."""
    if len(self._completed_thoughts) > 0:
        return self._completed_thoughts[len(self._completed_thoughts) - 1]
    return None