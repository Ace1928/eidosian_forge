from __future__ import annotations
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env
def _messages_to_prompt_dict(input_messages: List[BaseMessage]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Converts a list of LangChain Messages into a simple dict
    which is the message structure in Prem"""
    system_prompt: Optional[str] = None
    examples_and_messages: List[Dict[str, str]] = []
    for input_msg in input_messages:
        if isinstance(input_msg, SystemMessage):
            system_prompt = str(input_msg.content)
        elif isinstance(input_msg, HumanMessage):
            examples_and_messages.append({'role': 'user', 'content': str(input_msg.content)})
        elif isinstance(input_msg, AIMessage):
            examples_and_messages.append({'role': 'assistant', 'content': str(input_msg.content)})
        else:
            raise ChatPremAPIError('No such role explicitly exists')
    return (system_prompt, examples_and_messages)