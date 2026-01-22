from __future__ import annotations
from .base import BaseModel
from pydantic import Field, PrivateAttr
from ..utils.lazy import logger
from typing import Optional, Dict, Any, Union, List, TYPE_CHECKING
def add_app_url(self, allowed_origin: Optional[str]=None, callback: Optional[str]=None, web_origin: Optional[str]=None, allowed_logout_url: Optional[str]=None, verbose: Optional[bool]=None):
    """
        Adds to app urls
        """
    if allowed_origin is not None and allowed_origin not in self.allowed_origins:
        self.allowed_origins.append(allowed_origin)
        self._needs_update = True
        if verbose:
            logger.info(f'Added allowed origin: {allowed_origin}')
    if callback is not None and callback not in self.callbacks:
        self.callbacks.append(callback)
        self._needs_update = True
        if verbose:
            logger.info(f'Added callback: {callback}')
    if web_origin is not None and web_origin not in self.web_origins:
        self.web_origins.append(web_origin)
        self._needs_update = True
        if verbose:
            logger.info(f'Added web origin: {web_origin}')
    if allowed_logout_url is not None and allowed_logout_url not in self.allowed_logout_urls:
        self.allowed_logout_urls.append(allowed_logout_url)
        self._needs_update = True
        if verbose:
            logger.info(f'Added allowed logout url: {allowed_logout_url}')