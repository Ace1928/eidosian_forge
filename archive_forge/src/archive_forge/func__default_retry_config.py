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
def _default_retry_config() -> Retry:
    """Get the default retry configuration.

    If urllib3 version is 1.26 or greater, retry on all methods.

    Returns:
    -------
    Retry
        The default retry configuration.
    """
    retry_params = dict(total=3, status_forcelist=[502, 503, 504, 408, 425], backoff_factor=0.5, raise_on_redirect=False, raise_on_status=False)
    urllib3_version = importlib.metadata.version('urllib3')
    use_allowed_methods = tuple(map(int, urllib3_version.split('.'))) >= (1, 26)
    if use_allowed_methods:
        retry_params['allowed_methods'] = None
    return ls_utils.LangSmithRetry(**retry_params)