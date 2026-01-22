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
class CachedStFunctionWarning(StreamlitAPIWarning):

    def __init__(self, st_func_name, cached_func):
        msg = self._get_message(st_func_name, cached_func)
        super().__init__(msg)

    def _get_message(self, st_func_name, cached_func):
        args = {'st_func_name': '`st.%s()` or `st.write()`' % st_func_name, 'func_name': _get_cached_func_name_md(cached_func)}
        return ('\nYour script uses %(st_func_name)s to write to your Streamlit app from within\nsome cached code at %(func_name)s. This code will only be called when we detect\na cache "miss", which can lead to unexpected results.\n\nHow to fix this:\n* Move the %(st_func_name)s call outside %(func_name)s.\n* Or, if you know what you\'re doing, use `@st.cache(suppress_st_warning=True)`\nto suppress the warning.\n            ' % args).strip('\n')