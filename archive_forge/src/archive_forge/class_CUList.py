import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
@dataclass
class CUList(Component):
    """Component class for a list."""
    folder_id: float
    name: str
    content: Optional[str] = None
    due_date: Optional[int] = None
    due_date_time: Optional[bool] = None
    priority: Optional[int] = None
    assignee: Optional[int] = None
    status: Optional[str] = None

    @classmethod
    def from_data(cls, data: dict) -> 'CUList':
        return cls(folder_id=data['folder_id'], name=data['name'], content=data.get('content'), due_date=data.get('due_date'), due_date_time=data.get('due_date_time'), priority=data.get('priority'), assignee=data.get('assignee'), status=data.get('status'))