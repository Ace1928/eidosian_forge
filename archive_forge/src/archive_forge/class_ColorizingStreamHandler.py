import logging
import sys
from typing import Union
from rq.defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT
class ColorizingStreamHandler(logging.StreamHandler):
    levels = {logging.WARNING: yellow, logging.ERROR: red, logging.CRITICAL: red}

    def __init__(self, exclude=None, *args, **kwargs):
        self.exclude = exclude
        super().__init__(*args, **kwargs)

    @property
    def is_tty(self):
        isatty = getattr(self.stream, 'isatty', None)
        return isatty and isatty()

    def format(self, record):
        message = logging.StreamHandler.format(self, record)
        if self.is_tty:
            parts = message.split('\n', 1)
            parts[0] = ' '.join([parts[0].split(' ', 1)[0], parts[0].split(' ', 1)[1]])
            message = '\n'.join(parts)
        return message