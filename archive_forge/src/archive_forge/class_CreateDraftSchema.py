import base64
from email.message import EmailMessage
from typing import List, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.gmail.base import GmailBaseTool
class CreateDraftSchema(BaseModel):
    """Input for CreateDraftTool."""
    message: str = Field(..., description='The message to include in the draft.')
    to: List[str] = Field(..., description='The list of recipients.')
    subject: str = Field(..., description='The subject of the message.')
    cc: Optional[List[str]] = Field(None, description='The list of CC recipients.')
    bcc: Optional[List[str]] = Field(None, description='The list of BCC recipients.')