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
class _Cells:
    """
    Class which is basically a dict that allows us to push/pop frames of data.

    Python code objects are nested. In the following function:

        @st.cache()
        def func():
            production = [[x + y for x in range(3)] for y in range(5)]
            return production

    func.__code__ is a code object, and contains (inside
    func.__code__.co_consts) additional code objects for the list
    comprehensions. Those objects have their own co_freevars and co_cellvars.

    What we need to do as we're traversing this "tree" of code objects is to
    save each code object's vars, hash it, and then restore the original vars.
    """
    _cell_delete_obj = object()

    def __init__(self):
        self.values = {}
        self.stack = []
        self.frames = []

    def __repr__(self) -> str:
        return util.repr_(self)

    def _set(self, key, value):
        """
        Sets a value and saves the old value so it can be restored when
        we pop the frame. A sentinel object, _cell_delete_obj, indicates that
        the key was previously empty and should just be deleted.
        """
        self.stack.append((key, self.values.get(key, self._cell_delete_obj)))
        self.values[key] = value

    def pop(self):
        """Pop off the last frame we created, and restore all the old values."""
        idx = self.frames.pop()
        for key, val in self.stack[idx:]:
            if val is self._cell_delete_obj:
                del self.values[key]
            else:
                self.values[key] = val
        self.stack = self.stack[:idx]

    def push(self, code, func=None):
        """Create a new frame, and save all of `code`'s vars into it."""
        self.frames.append(len(self.stack))
        for var in code.co_cellvars:
            self._set(var, var)
        if code.co_freevars:
            if func is not None:
                assert len(code.co_freevars) == len(func.__closure__)
                for var, cell in zip(code.co_freevars, func.__closure__):
                    self._set(var, cell.cell_contents)
            else:
                for var in code.co_freevars:
                    self._set(var, var)