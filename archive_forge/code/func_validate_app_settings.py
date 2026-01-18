from __future__ import annotations
import os
import contextlib
from enum import Enum
from pathlib import Path
from pydantic import model_validator
from lazyops.utils.logs import Logger, null_logger
from lazyops.imports._pydantic import BaseAppSettings, BaseModel
from lazyops.libs.abcs.state import GlobalContext
from lazyops.libs.fastapi_utils.types.persistence import TemporaryData
from typing import List, Optional, Dict, Any, Callable, Union, Type, TYPE_CHECKING
from .types import AppEnv, get_app_env
@model_validator(mode='after')
def validate_app_settings(self):
    """
        Validates the app environment
        """
    from .ctx import ApplicationContextManager
    self.ctx = ApplicationContextManager.get_ctx(self.module_name, ingress_domain=self.ingress_domain, ingress_base=self.ingress_base)
    self.app_env = self.ctx.app_env
    if self.__class__.__name__.lower() == f'{self.__module__}settings':
        self.ctx.register_settings(self)
        from .lazy import register_module_settings
        register_module_settings(self.__module__, self)
    return self