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
class MessagesPlaceholder(BaseMessagePromptTemplate):
    """Prompt template that assumes variable is already list of messages.

    A placeholder which can be used to pass in a list of messages.

    Direct usage:

        .. code-block:: python

            from langchain_core.prompts import MessagesPlaceholder

            prompt = MessagesPlaceholder("history")
            prompt.format_messages() # raises KeyError

            prompt = MessagesPlaceholder("history", optional=True)
            prompt.format_messages() # returns empty list []

            prompt.format_messages(
                history=[
                    ("system", "You are an AI assistant."),
                    ("human", "Hello!"),
                ]
            )
            # -> [
            #     SystemMessage(content="You are an AI assistant."),
            #     HumanMessage(content="Hello!"),
            # ]

    Building a prompt with chat history:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    MessagesPlaceholder("history"),
                    ("human", "{question}")
                ]
            )
            prompt.invoke(
               {
                   "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
                   "question": "now multiply that by 4"
               }
            )
            # -> ChatPromptValue(messages=[
            #     SystemMessage(content="You are a helpful assistant."),
            #     HumanMessage(content="what's 5 + 2"),
            #     AIMessage(content="5 + 2 is 7"),
            #     HumanMessage(content="now multiply that by 4"),
            # ])
    """
    variable_name: str
    'Name of variable to use as messages.'
    optional: bool = False
    'If True format_messages can be called with no arguments and will return an empty \n        list. If False then a named argument with name `variable_name` must be passed \n        in, even if the value is an empty list.'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'prompts', 'chat']

    def __init__(self, variable_name: str, *, optional: bool=False, **kwargs: Any):
        super().__init__(variable_name=variable_name, optional=optional, **kwargs)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessage.
        """
        value = kwargs.get(self.variable_name, []) if self.optional else kwargs[self.variable_name]
        if not isinstance(value, list):
            raise ValueError(f'variable {self.variable_name} should be a list of base messages, got {value}')
        return convert_to_messages(value)

    @property
    def input_variables(self) -> List[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        return [self.variable_name] if not self.optional else []

    def pretty_repr(self, html: bool=False) -> str:
        var = '{' + self.variable_name + '}'
        if html:
            title = get_msg_title_repr('Messages Placeholder', bold=True)
            var = get_colored_text(var, 'yellow')
        else:
            title = get_msg_title_repr('Messages Placeholder')
        return f'{title}\n\n{var}'