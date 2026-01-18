from typing import Any, Dict, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
Validate that api key and python package exists in environment.