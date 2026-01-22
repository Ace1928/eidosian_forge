import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests import Request, Session
def _create_action_payload(self, instructions: str, params: Optional[Dict]=None, preview_only=False) -> Dict:
    """Create a payload for an action."""
    data = params if params else {}
    data.update({'instructions': instructions})
    if preview_only:
        data.update({'preview_only': True})
    return data