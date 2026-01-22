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
class AsyncCallbackManagerForChainRun(AsyncParentRunManager, ChainManagerMixin):
    """Async callback manager for chain run."""

    def get_sync(self) -> CallbackManagerForChainRun:
        """Get the equivalent sync RunManager.

        Returns:
            CallbackManagerForChainRun: The sync RunManager.
        """
        return CallbackManagerForChainRun(run_id=self.run_id, handlers=self.handlers, inheritable_handlers=self.inheritable_handlers, parent_run_id=self.parent_run_id, tags=self.tags, inheritable_tags=self.inheritable_tags, metadata=self.metadata, inheritable_metadata=self.inheritable_metadata)

    @shielded
    async def on_chain_end(self, outputs: Union[Dict[str, Any], Any], **kwargs: Any) -> None:
        """Run when chain ends running.

        Args:
            outputs (Union[Dict[str, Any], Any]): The outputs of the chain.
        """
        await ahandle_event(self.handlers, 'on_chain_end', 'ignore_chain', outputs, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, **kwargs)

    @shielded
    async def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when chain errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        await ahandle_event(self.handlers, 'on_chain_error', 'ignore_chain', error, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, **kwargs)

    @shielded
    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run when agent action is received.

        Args:
            action (AgentAction): The agent action.

        Returns:
            Any: The result of the callback.
        """
        await ahandle_event(self.handlers, 'on_agent_action', 'ignore_agent', action, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, **kwargs)

    @shielded
    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run when agent finish is received.

        Args:
            finish (AgentFinish): The agent finish.

        Returns:
            Any: The result of the callback.
        """
        await ahandle_event(self.handlers, 'on_agent_finish', 'ignore_agent', finish, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, **kwargs)