from typing import Dict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.github.prompt import (
from langchain_community.tools.github.tool import GitHubAction
from langchain_community.utilities.github import GitHubAPIWrapper
class ReadFile(BaseModel):
    """Schema for operations that require a file path as input."""
    formatted_filepath: str = Field(..., description='The full file path of the file you would like to read where the path must NOT start with a slash, e.g. `some_dir/my_file.py`.')