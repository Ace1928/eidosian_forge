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
def _create_template_from_message_type(message_type: str, template: Union[str, list], template_format: Literal['f-string', 'mustache']='f-string') -> BaseMessagePromptTemplate:
    """Create a message prompt template from a message type and template string.

    Args:
        message_type: str the type of the message template (e.g., "human", "ai", etc.)
        template: str the template string.

    Returns:
        a message prompt template of the appropriate type.
    """
    if message_type in ('human', 'user'):
        message: BaseMessagePromptTemplate = HumanMessagePromptTemplate.from_template(template, template_format=template_format)
    elif message_type in ('ai', 'assistant'):
        message = AIMessagePromptTemplate.from_template(cast(str, template), template_format=template_format)
    elif message_type == 'system':
        message = SystemMessagePromptTemplate.from_template(cast(str, template), template_format=template_format)
    elif message_type == 'placeholder':
        if isinstance(template, str):
            if template[0] != '{' or template[-1] != '}':
                raise ValueError(f'Invalid placeholder template: {template}. Expected a variable name surrounded by curly braces.')
            var_name = template[1:-1]
            message = MessagesPlaceholder(variable_name=var_name, optional=True)
        elif len(template) == 2 and isinstance(template[1], bool):
            var_name_wrapped, is_optional = template
            if not isinstance(var_name_wrapped, str):
                raise ValueError(f'Expected variable name to be a string. Got: {var_name_wrapped}')
            if var_name_wrapped[0] != '{' or var_name_wrapped[-1] != '}':
                raise ValueError(f'Invalid placeholder template: {var_name_wrapped}. Expected a variable name surrounded by curly braces.')
            var_name = var_name_wrapped[1:-1]
            message = MessagesPlaceholder(variable_name=var_name, optional=is_optional)
        else:
            raise ValueError(f'Unexpected arguments for placeholder message type. Expected either a single string variable name or a list of [variable_name: str, is_optional: bool]. Got: {template}')
    else:
        raise ValueError(f"Unexpected message type: {message_type}. Use one of 'human', 'user', 'ai', 'assistant', or 'system'.")
    return message