from typing import Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
Run query through GoogleSearchScholar and parse result