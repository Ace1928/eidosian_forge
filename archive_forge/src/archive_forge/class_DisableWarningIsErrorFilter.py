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
class DisableWarningIsErrorFilter(logging.Filter):
    """Disable WarningIsErrorFilter if this filter installed."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.skip_warningsiserror = True
        return True