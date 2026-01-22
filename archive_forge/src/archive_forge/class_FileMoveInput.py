import shutil
from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.tools.file_management.utils import (
class FileMoveInput(BaseModel):
    """Input for MoveFileTool."""
    source_path: str = Field(..., description='Path of the file to move')
    destination_path: str = Field(..., description='New path for the moved file')