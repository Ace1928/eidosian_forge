from __future__ import annotations
import contextlib
import functools
import hashlib
import inspect
import math
import os
import pickle
import shutil
import threading
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Final, Iterator, TypeVar, cast, overload
from cachetools import TTLCache
import streamlit as st
from streamlit import config, file_util, util
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.elements.spinner import spinner
from streamlit.error_util import handle_uncaught_app_exception
from streamlit.errors import StreamlitAPIWarning
from streamlit.logger import get_logger
from streamlit.runtime.caching import CACHE_DOCS_URL
from streamlit.runtime.caching.cache_type import CacheType, get_decorator_api_name
from streamlit.runtime.legacy_caching.hashing import (
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.stats import CacheStat, CacheStatsProvider
from streamlit.util import HASHLIB_KWARGS
def _get_output_hash(value: Any, func_or_code: Callable[..., Any], hash_funcs: HashFuncsDict | None) -> bytes:
    hasher = hashlib.new('md5', **HASHLIB_KWARGS)
    update_hash(value, hasher=hasher, hash_funcs=hash_funcs, hash_reason=HashReason.CACHING_FUNC_OUTPUT, hash_source=func_or_code)
    return hasher.digest()