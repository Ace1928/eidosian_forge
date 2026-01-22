import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
class ColoredLogRecord(logging.LogRecord):
    """Clones an existing logRecord, but reformats the levelname to add
    color, and the message to add bolding (where indicated by $BOLD
    and $RESET in the message).

    .. versionadded:: 2.2.0"""
    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7
    RESET_SEQ = '\x1b[0m'
    COLOR_SEQ = '\x1b[1;%dm'
    BOLD_SEQ = '\x1b[1m'
    LEVEL_COLORS = {'TRACE': MAGENTA, 'WARNING': YELLOW, 'INFO': GREEN, 'DEBUG': CYAN, 'CRITICAL': RED, 'ERROR': RED}

    @classmethod
    def _format_message(cls, message):
        return str(message).replace('$RESET', cls.RESET_SEQ).replace('$BOLD', cls.BOLD_SEQ)

    @classmethod
    def _format_levelname(cls, levelname):
        if levelname in cls.LEVEL_COLORS:
            return cls.COLOR_SEQ % (30 + cls.LEVEL_COLORS[levelname]) + levelname + cls.RESET_SEQ
        return levelname

    def __init__(self, logrecord):
        super().__init__(name=logrecord.name, level=logrecord.levelno, pathname=logrecord.pathname, lineno=logrecord.lineno, msg=logrecord.msg, args=logrecord.args, exc_info=logrecord.exc_info, func=logrecord.funcName, sinfo=logrecord.stack_info)
        self.levelname = self._format_levelname(self.levelname)
        self.msg = self._format_message(self.msg)