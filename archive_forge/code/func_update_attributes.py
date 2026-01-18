from __future__ import annotations
from pydantic import Field, model_validator, PrivateAttr
from lazyops.types.models import ConfigDict, schema_extra
from lazyops.libs import lazyload
from .base import BaseModel
from .common import UserType, ValidationMethod
from .user_roles import UserRole
from .user_session import UserSession
from .user_data import AZUserData
from .claims import UserJWTClaims, APIKeyJWTClaims
from .auth import AuthObject
from .security import Authorization, APIKey
from .errors import (
from ..utils.lazy import logger, ThreadPooler
from ..utils.helpers import parse_scopes, get_hashed_key
from ..utils.decoders import decode_token
from typing import Optional, List, Dict, Any, Union, Callable, Iterable, TYPE_CHECKING
def update_attributes(self, attributes: Dict[str, Any]):
    """
        Updates the attributes
        """
    for key, value in attributes.items():
        if isinstance(value, dict):
            self.attributes[key] = self.attributes.get(key, {})
            self.attributes[key].update(value)
        elif isinstance(value, list):
            self.attributes[key] = self.attributes.get(key, [])
            self.attributes[key].extend(value)
            self.attributes[key] = list(set(self.attributes[key]))
        else:
            self.attributes[key] = value