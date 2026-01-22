from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from tenacity import RetryCallState
class CallbackManagerMixin:
    """Mixin for callback manager."""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Any:
        """Run when LLM starts running.

        **ATTENTION**: This method is called for non-chat models (regular LLMs). If
            you're implementing a handler for a chat model,
            you should use on_chat_model_start instead.
        """

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Any:
        """Run when a chat model starts running.

        **ATTENTION**: This method is called for chat models. If you're implementing
            a handler for a non-chat model, you should use on_llm_start instead.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement `on_chat_model_start`')

    def on_retriever_start(self, serialized: Dict[str, Any], query: str, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Any:
        """Run when Retriever starts running."""

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Any:
        """Run when chain starts running."""

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, inputs: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Any:
        """Run when tool starts running."""