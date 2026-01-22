from typing import TYPE_CHECKING, Optional, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
class CloseSessionSchema(BaseModel):
    """Input for UpdateSessionTool."""
    sessionId: str = Field(..., description='The sessionId, received from one of the createSessions \n        or updateSessions run before')