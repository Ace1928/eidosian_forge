import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
class UncoloredLogRecord(logging.LogRecord):
    """Clones an existing logRecord, but reformats the message
    to remove $BOLD/$RESET markup.

    .. versionadded:: 2.2.0"""

    @classmethod
    def _format_message(cls, message):
        return str(message).replace('$RESET', '').replace('$BOLD', '')

    def __init__(self, logrecord):
        super().__init__(name=logrecord.name, level=logrecord.levelno, pathname=logrecord.pathname, lineno=logrecord.lineno, msg=logrecord.msg, args=logrecord.args, exc_info=logrecord.exc_info, func=logrecord.funcName, sinfo=logrecord.stack_info)
        self.msg = self._format_message(self.msg)