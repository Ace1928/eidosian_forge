from __future__ import annotations
import json
import logging
from typing import (
import aiohttp
import requests
from langchain_core.callbacks.manager import (
from langchain_core.language_models.chat_models import (
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.requests import Requests
class ChatDeepInfraException(Exception):
    """Exception raised when the DeepInfra API returns an error."""
    pass