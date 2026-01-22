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
def _is_langchain_hosted(url: str) -> bool:
    """Check if the URL is langchain hosted.

    Parameters
    ----------
    url : str
        The URL to check.

    Returns:
    -------
    bool
        True if the URL is langchain hosted, False otherwise.
    """
    try:
        netloc = urllib_parse.urlsplit(url).netloc.split(':')[0]
        return netloc.endswith('langchain.com')
    except Exception:
        return False