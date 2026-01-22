from __future__ import annotations
import contextlib
import copy
import math
import re
import types
from enum import Enum, EnumMeta, auto
from typing import (
from typing_extensions import TypeAlias, TypeGuard
import streamlit as st
from streamlit import config, errors
from streamlit import logger as _logger
from streamlit import string_util
from streamlit.errors import StreamlitAPIException
def _is_probably_plotly_dict(obj: object) -> TypeGuard[dict[str, Any]]:
    if not isinstance(obj, dict):
        return False
    if len(obj.keys()) == 0:
        return False
    if any((k not in ['config', 'data', 'frames', 'layout'] for k in obj.keys())):
        return False
    if any((_is_plotly_obj(v) for v in obj.values())):
        return True
    if any((_is_list_of_plotly_objs(v) for v in obj.values())):
        return True
    return False