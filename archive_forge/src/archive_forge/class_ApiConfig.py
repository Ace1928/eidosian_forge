from __future__ import annotations
import json
from typing import Optional, Type
import requests
import yaml
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
class ApiConfig(BaseModel):
    """API Configuration."""
    type: str
    url: str
    has_user_authentication: Optional[bool] = False