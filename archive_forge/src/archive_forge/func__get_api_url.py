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
def _get_api_url(api_url: Optional[str]) -> str:
    _api_url = api_url or cast(str, _get_env(('LANGSMITH_ENDPOINT', 'LANGCHAIN_ENDPOINT'), 'https://api.smith.langchain.com'))
    if not _api_url.strip():
        raise ls_utils.LangSmithUserError('LangSmith API URL cannot be empty')
    return _api_url.strip().strip('"').strip("'").rstrip('/')