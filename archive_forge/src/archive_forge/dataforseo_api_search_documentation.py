import base64
from typing import Dict, Optional
from urllib.parse import quote
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
Process response from DataForSEO SERP API.