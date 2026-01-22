import asyncio
import base64
import logging
import mimetypes
import os
from typing import Any, Dict, Optional, Type, Union
import requests
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
class NUASchema(BaseModel):
    """Input for Nuclia Understanding API.

    Attributes:
        action: Action to perform. Either `push` or `pull`.
        id: ID of the file to push or pull.
        path: Path to the file to push (needed only for `push` action).
        text: Text content to process (needed only for `push` action).
    """
    action: str = Field(..., description='Action to perform. Either `push` or `pull`.')
    id: str = Field(..., description='ID of the file to push or pull.')
    path: Optional[str] = Field(..., description='Path to the file to push (needed only for `push` action).')
    text: Optional[str] = Field(..., description='Text content to process (needed only for `push` action).')