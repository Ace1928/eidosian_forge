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
class SphinxLogRecord(logging.LogRecord):
    """Log record class supporting location"""
    prefix = ''
    location: Any = None

    def getMessage(self) -> str:
        message = super().getMessage()
        location = getattr(self, 'location', None)
        if location:
            message = '%s: %s%s' % (location, self.prefix, message)
        elif self.prefix not in message:
            message = self.prefix + message
        return message