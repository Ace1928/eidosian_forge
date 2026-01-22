from __future__ import annotations
import datetime
import warnings
from typing import Any, Dict, List, Optional, Type
from uuid import UUID
from langsmith.schemas import RunBase as BaseRunV2
from langsmith.schemas import RunTypeEnum as RunTypeEnumDep
from langchain_core._api import deprecated
from langchain_core.outputs import LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
@deprecated('0.1.0', removal='0.2.0')
class TracerSession(TracerSessionBase):
    """TracerSessionV1 schema for the V2 API."""
    id: UUID