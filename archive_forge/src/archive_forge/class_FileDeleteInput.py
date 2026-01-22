import os
from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.tools.file_management.utils import (
class FileDeleteInput(BaseModel):
    """Input for DeleteFileTool."""
    file_path: str = Field(..., description='Path of the file to delete')