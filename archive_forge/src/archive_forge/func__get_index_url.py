import logging
import os
import shutil
import subprocess
import sysconfig
import typing
import urllib.parse
from abc import ABC, abstractmethod
from functools import lru_cache
from os.path import commonprefix
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from pip._vendor.requests.auth import AuthBase, HTTPBasicAuth
from pip._vendor.requests.models import Request, Response
from pip._vendor.requests.utils import get_netrc_auth
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import (
from pip._internal.vcs.versioncontrol import AuthInfo
def _get_index_url(self, url: str) -> Optional[str]:
    """Return the original index URL matching the requested URL.

        Cached or dynamically generated credentials may work against
        the original index URL rather than just the netloc.

        The provided url should have had its username and password
        removed already. If the original index url had credentials then
        they will be included in the return value.

        Returns None if no matching index was found, or if --no-index
        was specified by the user.
        """
    if not url or not self.index_urls:
        return None
    url = remove_auth_from_url(url).rstrip('/') + '/'
    parsed_url = urllib.parse.urlsplit(url)
    candidates = []
    for index in self.index_urls:
        index = index.rstrip('/') + '/'
        parsed_index = urllib.parse.urlsplit(remove_auth_from_url(index))
        if parsed_url == parsed_index:
            return index
        if parsed_url.netloc != parsed_index.netloc:
            continue
        candidate = urllib.parse.urlsplit(index)
        candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda candidate: commonprefix([parsed_url.path, candidate.path]).rfind('/'))
    return urllib.parse.urlunsplit(candidates[0])