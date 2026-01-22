from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Literal, Sequence, cast
from typing_extensions import TypedDict
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
class ChatPromptValue(PromptValue):
    """Chat prompt value.

    A type of a prompt value that is built from messages.
    """
    messages: Sequence[BaseMessage]
    'List of messages.'

    def to_string(self) -> str:
        """Return prompt as string."""
        return get_buffer_string(self.messages)

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of messages."""
        return list(self.messages)

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'prompts', 'chat']