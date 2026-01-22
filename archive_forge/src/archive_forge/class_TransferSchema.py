import json
from typing import Optional, Type
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.ainetwork.base import AINBaseTool
class TransferSchema(BaseModel):
    """Schema for transfer operations."""
    address: str = Field(..., description='Address to transfer AIN to')
    amount: int = Field(..., description='Amount of AIN to transfer')