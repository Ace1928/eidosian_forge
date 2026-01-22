from __future__ import annotations
import os
import time
from pathlib import Path
from functools import lru_cache
from lazyops.utils.logs import logger as _logger, null_logger as _null_logger, Logger
from lazyops.imports._pydantic import BaseSettings
from lazyops.libs import lazyload
from lazyops.libs.proxyobj import ProxyObject
from lazyops.libs.abcs.configs.types import AppEnv
from lazyops.libs.fastapi_utils.types.persistence import TemporaryData
from pydantic import model_validator, computed_field, Field
from ..types.user_roles import UserRole
from ..utils.helpers import get_hashed_key, encrypt_key, decrypt_key, aencrypt_key, adecrypt_key, normalize_audience_name
from typing import List, Optional, Dict, Any, Union, overload, Callable, Tuple, TYPE_CHECKING
class AuthZeroContextObject:
    """
    The Auth Zero Context
    """
    pre_validate_hooks: Optional[List[Callable]] = []
    post_validate_hooks: Optional[List[Callable]] = []
    configured_validators: List[str] = []
    validation_order: Optional[List[str]] = ['session', 'api_key', 'token']

    def add_post_validate_hook(self, hook: Callable):
        """
        Adds a post validate hook
        """
        if self.post_validate_hooks is None:
            self.post_validate_hooks = []
        self.post_validate_hooks.append(hook)

    def add_pre_validate_hook(self, hook: Callable):
        """
        Adds a pre validate hook
        """
        if self.pre_validate_hooks is None:
            self.pre_validate_hooks = []
        self.pre_validate_hooks.append(hook)

    def get_validation_hooks(self) -> Tuple[List[Callable], List[Callable]]:
        """
        Returns the validation hooks
        """
        return (self.pre_validate_hooks.copy(), self.post_validate_hooks.copy())