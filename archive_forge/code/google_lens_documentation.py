from typing import Any, Dict, Optional, cast
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
Run query through Google Trends with Serpapi