import inspect
from typing import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import (
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain.chains import LLMChain
from langchain.output_parsers.ernie_functions import (
from langchain.utils.ernie_functions import convert_pydantic_to_ernie_function
def _get_python_function_arguments(function: Callable, arg_descriptions: dict) -> dict:
    """Get JsonSchema describing a Python functions arguments.

    Assumes all function arguments are of primitive types (int, float, str, bool) or
    are subclasses of pydantic.BaseModel.
    """
    properties = {}
    annotations = inspect.getfullargspec(function).annotations
    for arg, arg_type in annotations.items():
        if arg == 'return':
            continue
        if isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
            properties[arg] = arg_type.schema()
        elif arg_type.__name__ in PYTHON_TO_JSON_TYPES:
            properties[arg] = {'type': PYTHON_TO_JSON_TYPES[arg_type.__name__]}
        if arg in arg_descriptions:
            if arg not in properties:
                properties[arg] = {}
            properties[arg]['description'] = arg_descriptions[arg]
    return properties