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
def _parse_python_function_docstring(function: Callable) -> Tuple[str, dict]:
    """Parse the function and argument descriptions from the docstring of a function.

    Assumes the function docstring follows Google Python style guide.
    """
    docstring = inspect.getdoc(function)
    if docstring:
        docstring_blocks = docstring.split('\n\n')
        descriptors = []
        args_block = None
        past_descriptors = False
        for block in docstring_blocks:
            if block.startswith('Args:'):
                args_block = block
                break
            elif block.startswith('Returns:') or block.startswith('Example:'):
                past_descriptors = True
            elif not past_descriptors:
                descriptors.append(block)
            else:
                continue
        description = ' '.join(descriptors)
    else:
        description = ''
        args_block = None
    arg_descriptions = {}
    if args_block:
        arg = None
        for line in args_block.split('\n')[1:]:
            if ':' in line:
                arg, desc = line.split(':')
                arg_descriptions[arg.strip()] = desc.strip()
            elif arg:
                arg_descriptions[arg.strip()] += ' ' + line.strip()
    return (description, arg_descriptions)