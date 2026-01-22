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
class OnceFilter(logging.Filter):
    """Show the message only once."""

    def __init__(self, name: str='') -> None:
        super().__init__(name)
        self.messages: Dict[str, List] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        once = getattr(record, 'once', '')
        if not once:
            return True
        else:
            params = self.messages.setdefault(record.msg, [])
            if record.args in params:
                return False
            params.append(record.args)
            return True