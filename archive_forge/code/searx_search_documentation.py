import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import (
from langchain_core.utils import get_from_dict_or_env
Asynchronously query with json results.

        Uses aiohttp. See `results` for more info.
        