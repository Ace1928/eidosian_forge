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
def validate_app_ingress(self):
    """
        Validates the app ingress
        """
    if self.app_ingress is None:
        return
    if not self.app_ingress.startswith('http'):
        if 'localhost' in self.app_ingress or '127.0.0.1' in self.app_ingress or '0.0.0.0' in self.app_ingress:
            if not get_az_temp_data().has_logged('app_ingress_validate'):
                self.logger.warning('`app_ingress` is not using https. This is insecure and is not recommended')
            self.app_ingress = f'http://{self.app_ingress}'
        else:
            self.app_ingress = f'https://{self.app_ingress}'
    self.app_ingress = self.app_ingress.rstrip('/')