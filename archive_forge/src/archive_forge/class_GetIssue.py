from typing import Dict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.github.prompt import (
from langchain_community.tools.github.tool import GitHubAction
from langchain_community.utilities.github import GitHubAPIWrapper
class GetIssue(BaseModel):
    """Schema for operations that require an issue number as input."""
    issue_number: int = Field(0, description='Issue number as an integer, e.g. `42`')