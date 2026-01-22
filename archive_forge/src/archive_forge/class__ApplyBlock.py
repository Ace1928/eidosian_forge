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
class _ApplyBlock(_Node):

    def __init__(self, method: str, line: int, body: _Node) -> None:
        self.method = method
        self.line = line
        self.body = body

    def each_child(self) -> Iterable['_Node']:
        return (self.body,)

    def generate(self, writer: '_CodeWriter') -> None:
        method_name = '_tt_apply%d' % writer.apply_counter
        writer.apply_counter += 1
        writer.write_line('def %s():' % method_name, self.line)
        with writer.indent():
            writer.write_line('_tt_buffer = []', self.line)
            writer.write_line('_tt_append = _tt_buffer.append', self.line)
            self.body.generate(writer)
            writer.write_line("return _tt_utf8('').join(_tt_buffer)", self.line)
        writer.write_line('_tt_append(_tt_utf8(%s(%s())))' % (self.method, method_name), self.line)