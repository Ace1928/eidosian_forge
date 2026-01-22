from typing import Dict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.github.prompt import (
from langchain_community.tools.github.tool import GitHubAction
from langchain_community.utilities.github import GitHubAPIWrapper
class NoInput(BaseModel):
    """Schema for operations that do not require any input."""
    no_input: str = Field('', description='No input required, e.g. `` (empty string).')