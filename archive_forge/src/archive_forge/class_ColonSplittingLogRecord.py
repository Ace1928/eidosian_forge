import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
class ColonSplittingLogRecord(logging.LogRecord):
    """Clones an existing logRecord, but reformats the message field
    if it contains a colon.

    .. versionadded:: 2.2.0
    """

    def __init__(self, logrecord):
        try:
            parts = logrecord.msg.split(':', 1)
            if len(parts) == 2:
                new_msg = '[%-12s]%s' % (parts[0], parts[1])
            else:
                new_msg = parts[0]
        except Exception:
            new_msg = logrecord.msg
        super().__init__(name=logrecord.name, level=logrecord.levelno, pathname=logrecord.pathname, lineno=logrecord.lineno, msg=new_msg, args=logrecord.args, exc_info=logrecord.exc_info, func=logrecord.funcName, sinfo=logrecord.stack_info)