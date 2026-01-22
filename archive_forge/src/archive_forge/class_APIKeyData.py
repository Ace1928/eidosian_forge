import base64
import httpx
from lazyops.libs import lazyload
from typing import Optional, List, Dict, Any, Union, Iterable, Generator, AsyncGenerator
from .base import BaseModel
from .user_data import AZUserData
from .claims import APIKeyJWTClaims
class APIKeyData(BaseModel):
    """
    The stored API Key Data
    """
    user_data: AZUserData
    claims: APIKeyJWTClaims