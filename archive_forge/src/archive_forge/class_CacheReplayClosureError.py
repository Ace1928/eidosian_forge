from __future__ import annotations
import types
from typing import Any
from streamlit import type_util
from streamlit.errors import (
from streamlit.runtime.caching.cache_type import CacheType, get_decorator_api_name
class CacheReplayClosureError(StreamlitAPIException):

    def __init__(self, cache_type: CacheType, cached_func: types.FunctionType):
        func_name = get_cached_func_name_md(cached_func)
        decorator_name = get_decorator_api_name(cache_type)
        msg = f'\nWhile running {func_name}, a streamlit element is called on some layout block created outside the function.\nThis is incompatible with replaying the cached effect of that element, because the\nthe referenced block might not exist when the replay happens.\n\nHow to fix this:\n* Move the creation of $THING inside {func_name}.\n* Move the call to the streamlit element outside of {func_name}.\n* Remove the `@st.{decorator_name}` decorator from {func_name}.\n            '.strip('\n')
        super().__init__(msg)