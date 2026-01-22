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
class LastMessagesWriter:
    """Stream writer storing last 10 messages in memory to save trackback"""

    def __init__(self, app: 'Sphinx', stream: IO) -> None:
        self.app = app

    def write(self, data: str) -> None:
        self.app.messagelog.append(data)