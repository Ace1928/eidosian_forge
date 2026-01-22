from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
class LLMThought:
    """A thought in the LLM's thought stream."""

    def __init__(self, parent_container: DeltaGenerator, labeler: LLMThoughtLabeler, expanded: bool, collapse_on_complete: bool):
        """Initialize the LLMThought.

        Args:
            parent_container: The container we're writing into.
            labeler: The labeler to use for this thought.
            expanded: Whether the thought should be expanded by default.
            collapse_on_complete: Whether the thought should be collapsed.
        """
        self._container = MutableExpander(parent_container=parent_container, label=labeler.get_initial_label(), expanded=expanded)
        self._state = LLMThoughtState.THINKING
        self._llm_token_stream = ''
        self._llm_token_writer_idx: Optional[int] = None
        self._last_tool: Optional[ToolRecord] = None
        self._collapse_on_complete = collapse_on_complete
        self._labeler = labeler

    @property
    def container(self) -> MutableExpander:
        """The container we're writing into."""
        return self._container

    @property
    def last_tool(self) -> Optional[ToolRecord]:
        """The last tool executed by this thought"""
        return self._last_tool

    def _reset_llm_token_stream(self) -> None:
        self._llm_token_stream = ''
        self._llm_token_writer_idx = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str]) -> None:
        self._reset_llm_token_stream()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._llm_token_stream += _convert_newlines(token)
        self._llm_token_writer_idx = self._container.markdown(self._llm_token_stream, index=self._llm_token_writer_idx)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._reset_llm_token_stream()

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._container.markdown('**LLM encountered an error...**')
        self._container.exception(error)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self._state = LLMThoughtState.RUNNING_TOOL
        tool_name = serialized['name']
        self._last_tool = ToolRecord(name=tool_name, input_str=input_str)
        self._container.update(new_label=self._labeler.get_tool_label(self._last_tool, is_complete=False))

    def on_tool_end(self, output: Any, color: Optional[str]=None, observation_prefix: Optional[str]=None, llm_prefix: Optional[str]=None, **kwargs: Any) -> None:
        self._container.markdown(f'**{str(output)}**')

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self._container.markdown('**Tool encountered an error...**')
        self._container.exception(error)

    def on_agent_action(self, action: AgentAction, color: Optional[str]=None, **kwargs: Any) -> Any:
        pass

    def complete(self, final_label: Optional[str]=None) -> None:
        """Finish the thought."""
        if final_label is None and self._state == LLMThoughtState.RUNNING_TOOL:
            assert self._last_tool is not None, '_last_tool should never be null when _state == RUNNING_TOOL'
            final_label = self._labeler.get_tool_label(self._last_tool, is_complete=True)
        self._state = LLMThoughtState.COMPLETE
        if self._collapse_on_complete:
            self._container.update(new_label=final_label, new_expanded=False)
        else:
            self._container.update(new_label=final_label)

    def clear(self) -> None:
        """Remove the thought from the screen. A cleared thought can't be reused."""
        self._container.clear()