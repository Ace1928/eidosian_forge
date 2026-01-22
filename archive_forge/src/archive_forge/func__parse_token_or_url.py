from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def _parse_token_or_url(url_or_token: Union[str, uuid.UUID], api_url: str, num_parts: int=2) -> Tuple[str, str]:
    """Parse a public dataset URL or share token."""
    try:
        if isinstance(url_or_token, uuid.UUID) or uuid.UUID(url_or_token):
            return (api_url, str(url_or_token))
    except ValueError:
        pass
    parsed_url = urllib_parse.urlparse(str(url_or_token))
    path_parts = parsed_url.path.split('/')
    if len(path_parts) >= num_parts:
        token_uuid = path_parts[-num_parts]
    else:
        raise ls_utils.LangSmithUserError(f'Invalid public dataset URL: {url_or_token}')
    return (api_url, token_uuid)