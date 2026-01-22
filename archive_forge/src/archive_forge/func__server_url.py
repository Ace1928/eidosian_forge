from __future__ import annotations
import logging # isort:skip
import json
import os
import urllib
from typing import (
from uuid import uuid4
from ..core.types import ID
from ..util.serialization import make_id
from ..util.warnings import warn
from .state import curstate
def _server_url(url: str, port: int | None) -> str:
    """

    """
    port_ = f':{port}' if port is not None else ''
    if url.startswith('http'):
        return f'{url.rsplit(':', 1)[0]}{port_}{'/'}'
    else:
        return f'http://{url.split(':')[0]}{port_}{'/'}'