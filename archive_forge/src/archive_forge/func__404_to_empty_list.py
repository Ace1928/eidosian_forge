from __future__ import annotations
import inspect
from abc import ABC
from urllib.parse import urljoin
from fastapi import HTTPException
from lazyops.libs import lazyload
from lazyops.libs.proxyobj import ProxyObject
from lazyops.utils.helpers import timed_cache
from ..types.errors import InvalidOperationException
from ..types.auth import AuthZeroTokenAuth
from ..types.clients import AuthZeroClientObject
from ..utils.lazy import get_az_settings, logger
from .tokens import ClientCredentialsFlow
from typing import Optional, List, Dict, Any, Union
def _404_to_empty_list(self, response: Union['Response', 'AsyncResponse']) -> Optional[List]:
    """
        Handles the Response
        """
    try:
        return self._process_response(response)
    except HTTPException as e:
        if e.status_code == 404:
            return []
        raise e