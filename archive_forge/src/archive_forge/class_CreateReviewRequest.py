from typing import Dict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.github.prompt import (
from langchain_community.tools.github.tool import GitHubAction
from langchain_community.utilities.github import GitHubAPIWrapper
class CreateReviewRequest(BaseModel):
    """Schema for operations that require a username as input."""
    username: str = Field(..., description='GitHub username of the user being requested, e.g. `my_username`.')