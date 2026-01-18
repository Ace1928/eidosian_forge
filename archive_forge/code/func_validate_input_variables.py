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
@root_validator(pre=True)
def validate_input_variables(cls, values: dict) -> dict:
    """Validate input variables.

        If input_variables is not set, it will be set to the union of
        all input variables in the messages.

        Args:
            values: values to validate.

        Returns:
            Validated values.
        """
    messages = values['messages']
    input_vars = set()
    input_types: Dict[str, Any] = values.get('input_types', {})
    for message in messages:
        if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
            input_vars.update(message.input_variables)
        if isinstance(message, MessagesPlaceholder):
            if message.variable_name not in input_types:
                input_types[message.variable_name] = List[AnyMessage]
    if 'partial_variables' in values:
        input_vars = input_vars - set(values['partial_variables'])
    if 'input_variables' in values and values.get('validate_template'):
        if input_vars != set(values['input_variables']):
            raise ValueError(f'Got mismatched input_variables. Expected: {input_vars}. Got: {values['input_variables']}')
    else:
        values['input_variables'] = sorted(input_vars)
    values['input_types'] = input_types
    return values