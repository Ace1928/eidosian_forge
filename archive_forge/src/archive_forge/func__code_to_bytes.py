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
def _code_to_bytes(self, code, context: Context, func=None) -> bytes:
    h = hashlib.new('md5', **HASHLIB_KWARGS)
    self.update(h, code.co_code)
    consts = [n for n in code.co_consts if not isinstance(n, str) or not n.endswith('.<lambda>')]
    self.update(h, consts, context)
    context.cells.push(code, func=func)
    for ref in get_referenced_objects(code, context):
        self.update(h, ref, context)
    context.cells.pop()
    return h.digest()