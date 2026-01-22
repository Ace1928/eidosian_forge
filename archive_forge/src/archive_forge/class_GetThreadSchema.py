from typing import Dict, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.gmail.base import GmailBaseTool
class GetThreadSchema(BaseModel):
    """Input for GetMessageTool."""
    thread_id: str = Field(..., description='The thread ID.')