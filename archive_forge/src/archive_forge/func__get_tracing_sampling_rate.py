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
def _get_tracing_sampling_rate() -> float | None:
    """Get the tracing sampling rate.

    Returns:
    -------
    float
        The tracing sampling rate.
    """
    sampling_rate_str = ls_utils.get_env_var('TRACING_SAMPLING_RATE')
    if sampling_rate_str is None:
        return None
    sampling_rate = float(sampling_rate_str)
    if sampling_rate < 0 or sampling_rate > 1:
        raise ls_utils.LangSmithUserError(f'LANGSMITH_TRACING_SAMPLING_RATE must be between 0 and 1 if set. Got: {sampling_rate}')
    return sampling_rate