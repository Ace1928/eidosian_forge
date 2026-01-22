import builtins
import json
from typing import Optional, Type, Union
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.ainetwork.base import AINBaseTool, OperationType
class ValueSchema(BaseModel):
    """Schema for value operations."""
    type: OperationType = Field(...)
    path: str = Field(..., description='Blockchain reference path')
    value: Optional[Union[int, str, float, dict]] = Field(None, description='Value to be set at the path')