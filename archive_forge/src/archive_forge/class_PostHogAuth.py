from __future__ import annotations
import gc
import niquests
from urllib.parse import urljoin
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Union, Literal, TYPE_CHECKING
class PostHogAuth(niquests.auth.BearerTokenAuth):
    """
    The PostHog Auth
    """

    def __init__(self, api_key: str):
        """
        Initializes the PostHog Auth
        """
        self.api_key = api_key
        super().__init__(token=api_key)

    def __call__(self, r: 'niquests.Request') -> 'niquests.Request':
        """
        The Call Method
        """
        r.headers['Authorization'] = f'Bearer {self.api_key}'
        r.headers['Accept'] = 'application/json'
        r.headers['Content-Type'] = 'application/json'
        return r