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
class SafeEncodingWriter:
    """Stream writer which ignores UnicodeEncodeError silently"""

    def __init__(self, stream: IO) -> None:
        self.stream = stream
        self.encoding = getattr(stream, 'encoding', 'ascii') or 'ascii'

    def write(self, data: str) -> None:
        try:
            self.stream.write(data)
        except UnicodeEncodeError:
            self.stream.write(data.encode(self.encoding, 'replace').decode(self.encoding))

    def flush(self) -> None:
        if hasattr(self.stream, 'flush'):
            self.stream.flush()