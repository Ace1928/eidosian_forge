from __future__ import annotations
import time
from .base import BaseModel
from pydantic import Field, model_validator
from typing import Dict, Optional, List, Any
class UserDataBase(BaseModel):
    """
    Base User Data
    """
    username: Optional[str] = None
    family_name: Optional[str] = None
    given_name: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    email_verified: Optional[bool] = None
    nickname: Optional[str] = None
    phone_number: Optional[str] = None
    phone_verified: Optional[bool] = None
    picture: Optional[str] = None
    user_metadata: Optional[Dict[Any, Any]] = None
    app_metadata: Optional[Dict[Any, Any]] = None