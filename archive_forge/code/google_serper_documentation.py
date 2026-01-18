from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
from typing_extensions import Literal
Run query through GoogleSearch and parse result async.