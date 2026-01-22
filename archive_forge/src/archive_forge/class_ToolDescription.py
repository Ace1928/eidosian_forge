from typing import Literal, Optional, Type, TypedDict
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs
class ToolDescription(TypedDict):
    """Representation of a callable function to the Ernie API."""
    type: Literal['function']
    function: FunctionDescription