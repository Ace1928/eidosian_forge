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
class WarningIsErrorFilter(logging.Filter):
    """Raise exception if warning emitted."""

    def __init__(self, app: 'Sphinx') -> None:
        self.app = app
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, 'skip_warningsiserror', False):
            return True
        elif self.app.warningiserror:
            location = getattr(record, 'location', '')
            try:
                message = record.msg % record.args
            except (TypeError, ValueError):
                message = record.msg
            if location:
                exc = SphinxWarning(location + ':' + str(message))
            else:
                exc = SphinxWarning(message)
            if record.exc_info is not None:
                raise exc from record.exc_info[1]
            else:
                raise exc
        else:
            return True