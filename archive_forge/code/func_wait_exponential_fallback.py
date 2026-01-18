from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, final
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def wait_exponential_fallback(multiplier: float=1, min: float=0, max: float=float('inf')) -> None:
    return None