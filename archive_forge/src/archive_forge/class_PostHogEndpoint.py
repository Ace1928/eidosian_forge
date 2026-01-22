from __future__ import annotations
import gc
import niquests
from urllib.parse import urljoin
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Union, Literal, TYPE_CHECKING
class PostHogEndpoint(BaseModel):
    endpoint: str

    def get_url(self, *paths: str) -> str:
        """
        Returns the URL
        """
        return urljoin(self.endpoint, '/'.join(paths)).rstrip('/')

    @property
    def capture(self) -> str:
        """
        Returns the Capture URL
        """
        return self.get_url('capture')

    @property
    def batch(self) -> str:
        """
        Returns the Batch URL
        """
        return self.get_url('batch')

    @property
    def identify(self) -> str:
        """
        Returns the Identify URL
        """
        return self.get_url('identify')