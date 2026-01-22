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
class _ControlBlock(_Node):

    def __init__(self, statement: str, line: int, body: _Node) -> None:
        self.statement = statement
        self.line = line
        self.body = body

    def each_child(self) -> Iterable[_Node]:
        return (self.body,)

    def generate(self, writer: '_CodeWriter') -> None:
        writer.write_line('%s:' % self.statement, self.line)
        with writer.indent():
            self.body.generate(writer)
            writer.write_line('pass', self.line)