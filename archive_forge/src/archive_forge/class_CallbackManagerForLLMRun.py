from __future__ import annotations
import asyncio
import functools
import logging
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import copy_context
from typing import (
from uuid import UUID
from langsmith.run_helpers import get_run_tree_context
from tenacity import RetryCallState
from langchain_core.callbacks.base import (
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils.env import env_var_is_set
class CallbackManagerForLLMRun(RunManager, LLMManagerMixin):
    """Callback manager for LLM run."""

    def on_llm_new_token(self, token: str, *, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]]=None, **kwargs: Any) -> None:
        """Run when LLM generates a new token.

        Args:
            token (str): The new token.
        """
        handle_event(self.handlers, 'on_llm_new_token', 'ignore_llm', token=token, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, chunk=chunk, **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The LLM result.
        """
        handle_event(self.handlers, 'on_llm_end', 'ignore_llm', response, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, **kwargs)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when LLM errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
            kwargs (Any): Additional keyword arguments.
                - response (LLMResult): The response which was generated before
                    the error occurred.
        """
        handle_event(self.handlers, 'on_llm_error', 'ignore_llm', error, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, **kwargs)