from __future__ import annotations
from pydantic import model_validator, PrivateAttr
from lazyops.types import BaseSettings
from lazyops.libs.proxyobj import ProxyObject
from typing import Optional
@model_validator(mode='after')
def validate_posthog_config(self):
    """
        Validates the Posthog Configuration
        """
    self.update_enabled()
    return self