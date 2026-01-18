from __future__ import annotations
import warnings
from abc import ABC
from string import Formatter
from typing import Any, Callable, Dict, List, Set, Tuple, Type
import langchain_core.utils.mustache as mustache
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, create_model
from langchain_core.utils import get_colored_text
from langchain_core.utils.formatting import formatter
from langchain_core.utils.interactive_env import is_interactive_env
def validate_jinja2(template: str, input_variables: List[str]) -> None:
    """
    Validate that the input variables are valid for the template.
    Issues a warning if missing or extra variables are found.

    Args:
        template: The template string.
        input_variables: The input variables.
    """
    input_variables_set = set(input_variables)
    valid_variables = _get_jinja2_variables_from_template(template)
    missing_variables = valid_variables - input_variables_set
    extra_variables = input_variables_set - valid_variables
    warning_message = ''
    if missing_variables:
        warning_message += f'Missing variables: {missing_variables} '
    if extra_variables:
        warning_message += f'Extra variables: {extra_variables}'
    if warning_message:
        warnings.warn(warning_message.strip())