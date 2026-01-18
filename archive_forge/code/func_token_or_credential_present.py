from __future__ import annotations
import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union
import aiohttp
import requests
from aiohttp import ServerTimeoutError
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator, validator
from requests.exceptions import Timeout
@root_validator(pre=True, allow_reuse=True)
def token_or_credential_present(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that at least one of token and credentials is present."""
    if 'token' in values or 'credential' in values:
        return values
    raise ValueError('Please provide either a credential or a token.')