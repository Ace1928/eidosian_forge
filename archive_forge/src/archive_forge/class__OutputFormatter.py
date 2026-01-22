import json
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union
from langchain_core.output_parsers import (
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import (
from langchain.output_parsers import (
class _OutputFormatter(BaseModel):
    """Output formatter. Should always be used to format your response to the user."""
    output: output_schema