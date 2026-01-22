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
class _CodeWriter(object):

    def __init__(self, file: TextIO, named_blocks: Dict[str, _NamedBlock], loader: Optional[BaseLoader], current_template: Template) -> None:
        self.file = file
        self.named_blocks = named_blocks
        self.loader = loader
        self.current_template = current_template
        self.apply_counter = 0
        self.include_stack = []
        self._indent = 0

    def indent_size(self) -> int:
        return self._indent

    def indent(self) -> 'ContextManager':

        class Indenter(object):

            def __enter__(_) -> '_CodeWriter':
                self._indent += 1
                return self

            def __exit__(_, *args: Any) -> None:
                assert self._indent > 0
                self._indent -= 1
        return Indenter()

    def include(self, template: Template, line: int) -> 'ContextManager':
        self.include_stack.append((self.current_template, line))
        self.current_template = template

        class IncludeTemplate(object):

            def __enter__(_) -> '_CodeWriter':
                return self

            def __exit__(_, *args: Any) -> None:
                self.current_template = self.include_stack.pop()[0]
        return IncludeTemplate()

    def write_line(self, line: str, line_number: int, indent: Optional[int]=None) -> None:
        if indent is None:
            indent = self._indent
        line_comment = '  # %s:%d' % (self.current_template.name, line_number)
        if self.include_stack:
            ancestors = ['%s:%d' % (tmpl.name, lineno) for tmpl, lineno in self.include_stack]
            line_comment += ' (via %s)' % ', '.join(reversed(ancestors))
        print('    ' * indent + line + line_comment, file=self.file)