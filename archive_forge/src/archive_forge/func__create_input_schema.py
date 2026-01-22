import asyncio
from functools import partial
from typing import Any, Dict, List, Optional, Type
from langchain_core.callbacks.manager import (
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator
from langchain_core.tools import BaseTool
from langchain_community.tools.connery.models import Action, Parameter
@classmethod
def _create_input_schema(cls, inputParameters: List[Parameter]) -> Type[BaseModel]:
    """
        Creates an input schema for a Connery Action Tool
        based on the input parameters of the Connery Action.
        Parameters:
            inputParameters: List of input parameters of the Connery Action.
        Returns:
            Type[BaseModel]: The input schema for the Connery Action Tool.
        """
    dynamic_input_fields: Dict[str, Any] = {}
    for param in inputParameters:
        default = ... if param.validation and param.validation.required else None
        title = param.title
        description = param.title + (': ' + param.description if param.description else '')
        type = param.type
        dynamic_input_fields[param.key] = (str, Field(default, title=title, description=description, type=type))
    InputModel = create_model('InputSchema', **dynamic_input_fields)
    return InputModel