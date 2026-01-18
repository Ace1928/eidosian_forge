from typing import Any, Dict, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
Use aiohttp to send request to SearchApi API and return results async.