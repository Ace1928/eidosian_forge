from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def _import_tiktoken() -> Any:
    """Import tiktoken."""
    try:
        import tiktoken
    except ImportError:
        raise ImportError('tiktoken is not installed. Please install it with `pip install tiktoken`')
    return tiktoken