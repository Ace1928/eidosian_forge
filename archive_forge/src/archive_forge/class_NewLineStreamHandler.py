import logging
import logging.handlers
from collections import defaultdict
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Node
from docutils.utils import get_source_line
from sphinx.errors import SphinxWarning
from sphinx.util.console import colorize
from sphinx.util.osutil import abspath
class NewLineStreamHandler(logging.StreamHandler):
    """StreamHandler which switches line terminator by record.nonl flag."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.acquire()
            if getattr(record, 'nonl', False):
                self.terminator = ''
            super().emit(record)
        finally:
            self.terminator = '\n'
            self.release()