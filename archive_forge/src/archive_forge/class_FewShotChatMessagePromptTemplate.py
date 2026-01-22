from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts.chat import (
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
class FewShotChatMessagePromptTemplate(BaseChatPromptTemplate, _FewShotPromptTemplateMixin):
    """Chat prompt template that supports few-shot examples.

    The high level structure of produced by this prompt template is a list of messages
    consisting of prefix message(s), example message(s), and suffix message(s).

    This structure enables creating a conversation with intermediate examples like:

        System: You are a helpful AI Assistant
        Human: What is 2+2?
        AI: 4
        Human: What is 2+3?
        AI: 5
        Human: What is 4+4?

    This prompt template can be used to generate a fixed list of examples or else
    to dynamically select examples based on the input.

    Examples:

        Prompt template with a fixed list of examples (matching the sample
        conversation above):

        .. code-block:: python

            from langchain_core.prompts import (
                FewShotChatMessagePromptTemplate,
                ChatPromptTemplate
            )

            examples = [
                {"input": "2+2", "output": "4"},
                {"input": "2+3", "output": "5"},
            ]

            example_prompt = ChatPromptTemplate.from_messages(
                [('human', '{input}'), ('ai', '{output}')]
            )

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                examples=examples,
                # This is a prompt template used to format each individual example.
                example_prompt=example_prompt,
            )

            final_prompt = ChatPromptTemplate.from_messages(
                [
                    ('system', 'You are a helpful AI Assistant'),
                    few_shot_prompt,
                    ('human', '{input}'),
                ]
            )
            final_prompt.format(input="What is 4+4?")

        Prompt template with dynamically selected examples:

        .. code-block:: python

            from langchain_core.prompts import SemanticSimilarityExampleSelector
            from langchain_core.embeddings import OpenAIEmbeddings
            from langchain_core.vectorstores import Chroma

            examples = [
                {"input": "2+2", "output": "4"},
                {"input": "2+3", "output": "5"},
                {"input": "2+4", "output": "6"},
                # ...
            ]

            to_vectorize = [
                " ".join(example.values())
                for example in examples
            ]
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_texts(
                to_vectorize, embeddings, metadatas=examples
            )
            example_selector = SemanticSimilarityExampleSelector(
                vectorstore=vectorstore
            )

            from langchain_core import SystemMessage
            from langchain_core.prompts import HumanMessagePromptTemplate
            from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                # Which variable(s) will be passed to the example selector.
                input_variables=["input"],
                example_selector=example_selector,
                # Define how each example will be formatted.
                # In this case, each example will become 2 messages:
                # 1 human, and 1 AI
                example_prompt=(
                    HumanMessagePromptTemplate.from_template("{input}")
                    + AIMessagePromptTemplate.from_template("{output}")
                ),
            )
            # Define the overall prompt.
            final_prompt = (
                SystemMessagePromptTemplate.from_template(
                    "You are a helpful AI Assistant"
                )
                + few_shot_prompt
                + HumanMessagePromptTemplate.from_template("{input}")
            )
            # Show the prompt
            print(final_prompt.format_messages(input="What's 3+3?"))  # noqa: T201

            # Use within an LLM
            from langchain_core.chat_models import ChatAnthropic
            chain = final_prompt | ChatAnthropic()
            chain.invoke({"input": "What's 3+3?"})
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether or not the class is serializable."""
        return False
    input_variables: List[str] = Field(default_factory=list)
    'A list of the names of the variables the prompt template will use\n    to pass to the example_selector, if provided.'
    example_prompt: Union[BaseMessagePromptTemplate, BaseChatPromptTemplate]
    'The class to format each example.'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format kwargs into a list of messages.

        Args:
            **kwargs: keyword arguments to use for filling in templates in messages.

        Returns:
            A list of formatted messages with all template variables filled in.
        """
        examples = self._get_examples(**kwargs)
        examples = [{k: e[k] for k in self.example_prompt.input_variables} for e in examples]
        messages = [message for example in examples for message in self.example_prompt.format_messages(**example)]
        return messages

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format kwargs into a list of messages.

        Args:
            **kwargs: keyword arguments to use for filling in templates in messages.

        Returns:
            A list of formatted messages with all template variables filled in.
        """
        examples = await self._aget_examples(**kwargs)
        examples = [{k: e[k] for k in self.example_prompt.input_variables} for e in examples]
        messages = [message for example in examples for message in await self.example_prompt.aformat_messages(**example)]
        return messages

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with inputs generating a string.

        Use this method to generate a string representation of a prompt consisting
        of chat messages.

        Useful for feeding into a string based completion language model or debugging.

        Args:
            **kwargs: keyword arguments to use for formatting.

        Returns:
            A string representation of the prompt
        """
        messages = self.format_messages(**kwargs)
        return get_buffer_string(messages)

    async def aformat(self, **kwargs: Any) -> str:
        messages = await self.aformat_messages(**kwargs)
        return get_buffer_string(messages)

    def pretty_repr(self, html: bool=False) -> str:
        raise NotImplementedError()