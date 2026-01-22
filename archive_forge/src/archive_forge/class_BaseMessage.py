from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts
from langchain_core.utils.interactive_env import is_interactive_env
class BaseMessage(Serializable):
    """Base abstract Message class.

    Messages are the inputs and outputs of ChatModels.
    """
    content: Union[str, List[Union[str, Dict]]]
    'The string contents of the message.'
    additional_kwargs: dict = Field(default_factory=dict)
    'Reserved for additional payload data associated with the message.\n    \n    For example, for a message from an AI, this could include tool calls.'
    response_metadata: dict = Field(default_factory=dict)
    'Response metadata. For example: response headers, logprobs, token counts.'
    type: str
    name: Optional[str] = None
    id: Optional[str] = None
    'An optional unique identifier for the message. This should ideally be\n    provided by the provider/model which created the message.'

    class Config:
        extra = Extra.allow

    def __init__(self, content: Union[str, List[Union[str, Dict]]], **kwargs: Any) -> None:
        """Pass in content as positional arg."""
        return super().__init__(content=content, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']

    def __add__(self, other: Any) -> ChatPromptTemplate:
        from langchain_core.prompts.chat import ChatPromptTemplate
        prompt = ChatPromptTemplate(messages=[self])
        return prompt + other

    def pretty_repr(self, html: bool=False) -> str:
        title = get_msg_title_repr(self.type.title() + ' Message', bold=html)
        if self.name is not None:
            title += f'\nName: {self.name}'
        return f'{title}\n\n{self.content}'

    def pretty_print(self) -> None:
        print(self.pretty_repr(html=is_interactive_env()))