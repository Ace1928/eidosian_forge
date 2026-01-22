from __future__ import annotations
import types
from typing import Any
from streamlit import type_util
from streamlit.errors import (
from streamlit.runtime.caching.cache_type import CacheType, get_decorator_api_name
class UnserializableReturnValueError(MarkdownFormattedException):

    def __init__(self, func: types.FunctionType, return_value: types.FunctionType):
        MarkdownFormattedException.__init__(self, f'\n            Cannot serialize the return value (of type {get_return_value_type(return_value)}) in {get_cached_func_name_md(func)}.\n            `st.cache_data` uses [pickle](https://docs.python.org/3/library/pickle.html) to\n            serialize the function’s return value and safely store it in the cache without mutating the original object. Please convert the return value to a pickle-serializable type.\n            If you want to cache unserializable objects such as database connections or Tensorflow\n            sessions, use `st.cache_resource` instead (see [our docs]({CACHE_DOCS_URL}) for differences).')