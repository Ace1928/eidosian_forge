from __future__ import annotations
import os
import threading
from copy import deepcopy
from typing import (
from blinker import Signal
import streamlit as st
import streamlit.watcher.path_watcher
from streamlit import file_util, runtime
from streamlit.logger import get_logger
@staticmethod
def _maybe_wrap_in_attr_dict(value) -> Any:
    if not isinstance(value, Mapping):
        return value
    else:
        return AttrDict(value)