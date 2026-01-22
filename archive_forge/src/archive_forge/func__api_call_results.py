from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, final
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _api_call_results(self, search_term: str) -> dict:
    """Call the NutritionAI API and return the results."""
    rsp = self._http_get({'term': search_term, **self.more_kwargs})
    if not rsp:
        raise ValueError('Could not get NutritionAI API results')
    rsp.raise_for_status()
    return rsp.json()