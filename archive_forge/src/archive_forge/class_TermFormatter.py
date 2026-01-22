from __future__ import annotations
import logging
from typing import (
import param
from ..io.resources import CDN_DIST
from ..io.state import state
from ..layout import Card, HSpacer, Row
from ..reactive import ReactiveHTML
from .terminal import Terminal
class TermFormatter(logging.Formatter):

    def __init__(self, *args, only_last=True, **kwargs):
        """
        Standard logging.Formatter with the default option of prompting
        only the last stack. Does not cache exc_text.

        Parameters
        ----------
        only_last : BOOLEAN, optional
            Whether the full stack trace or only the last file should be shown.
            The default is True.
        """
        super().__init__(*args, **kwargs)
        self.only_last = only_last

    def format(self, record):
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        s = self.formatMessage(record)
        exc_text = None
        if record.exc_info:
            exc_text = super().formatException(record.exc_info)
            last = exc_text.rfind('File')
            if last > 0 and self.only_last:
                exc_text = exc_text[last:]
        if exc_text:
            if s[-1:] != '\n':
                s = s + '\n'
            s = s + exc_text
        if record.stack_info:
            if s[-1:] != '\n':
                s = s + '\n'
            s = s + self.formatStack(record.stack_info)
        return s