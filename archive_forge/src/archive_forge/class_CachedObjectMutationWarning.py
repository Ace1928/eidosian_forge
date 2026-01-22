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
class CachedObjectMutationWarning(StreamlitAPIWarning):

    def __init__(self, orig_exc):
        msg = self._get_message(orig_exc)
        super().__init__(msg)

    def _get_message(self, orig_exc):
        return ("\nReturn value of %(func_name)s was mutated between runs.\n\nBy default, Streamlit's cache should be treated as immutable, or it may behave\nin unexpected ways. You received this warning because Streamlit detected\nthat an object returned by %(func_name)s was mutated outside of %(func_name)s.\n\nHow to fix this:\n* If you did not mean to mutate that return value:\n  - If possible, inspect your code to find and remove that mutation.\n  - Otherwise, you could also clone the returned value so you can freely\n    mutate it.\n* If you actually meant to mutate the return value and know the consequences of\ndoing so, annotate the function with `@st.cache(allow_output_mutation=True)`.\n\nFor more information and detailed solutions check out [our documentation.]\n(https://docs.streamlit.io/library/advanced-features/caching)\n            " % {'func_name': orig_exc.cached_func_name}).strip('\n')