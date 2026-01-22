from __future__ import annotations
import collections
import enum
import functools
import hashlib
import inspect
import io
import os
import pickle
import sys
import tempfile
import textwrap
import threading
import weakref
from typing import Any, Callable, Dict, Pattern, Type, Union
from streamlit import config, file_util, type_util, util
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
from streamlit.folder_black_list import FolderBlackList
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.util import HASHLIB_KWARGS
from, try looking at the hash chain below for an object that you do recognize,
from, try looking at the hash chain below for an object that you do recognize,
class _HashStacks:
    """Stacks of what has been hashed, with at most 1 stack per thread."""

    def __init__(self):
        self._stacks: weakref.WeakKeyDictionary[threading.Thread, _HashStack] = weakref.WeakKeyDictionary()

    def __repr__(self) -> str:
        return util.repr_(self)

    @property
    def current(self) -> _HashStack:
        current_thread = threading.current_thread()
        stack = self._stacks.get(current_thread, None)
        if stack is None:
            stack = _HashStack()
            self._stacks[current_thread] = stack
        return stack