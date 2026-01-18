import json
from typing import Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
Get results from the Metaphor Search API asynchronously.