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
def _get_python_function_required_args(function: Callable) -> List[str]:
    """Get the required arguments for a Python function."""
    spec = inspect.getfullargspec(function)
    required = spec.args[:-len(spec.defaults)] if spec.defaults else spec.args
    required += [k for k in spec.kwonlyargs if k not in (spec.kwonlydefaults or {})]
    is_class = type(function) is type
    if is_class and required[0] == 'self':
        required = required[1:]
    return required