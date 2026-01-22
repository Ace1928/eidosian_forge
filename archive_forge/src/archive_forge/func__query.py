import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
def _query(self, url: str, query_data: Optional[Dict[str, Any]]=None, **kwargs: Any) -> None:
    query_data = query_data or {}
    result = self._gl.http_request('get', url, query_data=query_data, **kwargs)
    try:
        next_url = result.links['next']['url']
    except KeyError:
        next_url = None
    self._next_url = self._gl._check_url(next_url)
    self._current_page: Optional[str] = result.headers.get('X-Page')
    self._prev_page: Optional[str] = result.headers.get('X-Prev-Page')
    self._next_page: Optional[str] = result.headers.get('X-Next-Page')
    self._per_page: Optional[str] = result.headers.get('X-Per-Page')
    self._total_pages: Optional[str] = result.headers.get('X-Total-Pages')
    self._total: Optional[str] = result.headers.get('X-Total')
    try:
        self._data: List[Dict[str, Any]] = result.json()
    except Exception as e:
        raise gitlab.exceptions.GitlabParsingError(error_message='Failed to parse the server message') from e
    self._current = 0