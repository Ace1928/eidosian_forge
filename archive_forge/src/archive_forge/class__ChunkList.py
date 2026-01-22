import datetime
from io import StringIO
import linecache
import os.path
import posixpath
import re
import threading
from tornado import escape
from tornado.log import app_log
from tornado.util import ObjectDict, exec_in, unicode_type
from typing import Any, Union, Callable, List, Dict, Iterable, Optional, TextIO
import typing
class _ChunkList(_Node):

    def __init__(self, chunks: List[_Node]) -> None:
        self.chunks = chunks

    def generate(self, writer: '_CodeWriter') -> None:
        for chunk in self.chunks:
            chunk.generate(writer)

    def each_child(self) -> Iterable['_Node']:
        return self.chunks