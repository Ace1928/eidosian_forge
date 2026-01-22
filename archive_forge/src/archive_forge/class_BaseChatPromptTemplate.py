from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
from langchain_core._api import deprecated
from langchain_core.load import Serializable
from langchain_core.messages import (
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompt_values import ChatPromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import StringPromptTemplate, get_template_variables
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_colored_text
from langchain_core.utils.interactive_env import is_interactive_env
class BaseChatPromptTemplate(BasePromptTemplate, ABC):
    """Base class for chat prompt templates."""

    @property
    def lc_attributes(self) -> Dict:
        """
        Return a list of attribute names that should be included in the
        serialized kwargs. These attributes must be accepted by the
        constructor.
        """
        return {'input_variables': self.input_variables}

    def format(self, **kwargs: Any) -> str:
        """Format the chat template into a string.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            formatted string
        """
        return self.format_prompt(**kwargs).to_string()

    async def aformat(self, **kwargs: Any) -> str:
        """Format the chat template into a string.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            formatted string
        """
        return (await self.aformat_prompt(**kwargs)).to_string()

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """
        Format prompt. Should return a PromptValue.
        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            PromptValue.
        """
        messages = self.format_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        messages = await self.aformat_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format kwargs into a list of messages."""

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format kwargs into a list of messages."""
        return self.format_messages(**kwargs)

    def pretty_repr(self, html: bool=False) -> str:
        """Human-readable representation."""
        raise NotImplementedError

    def pretty_print(self) -> None:
        print(self.pretty_repr(html=is_interactive_env()))