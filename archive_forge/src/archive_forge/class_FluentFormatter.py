import datetime
import debtcollector
import functools
import io
import itertools
import logging
import logging.config
import logging.handlers
import re
import socket
import sys
import traceback
from dateutil import tz
from oslo_context import context as context_utils
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
class FluentFormatter(logging.Formatter):
    """A formatter for fluentd.

    format() returns dict, not string.
    It expects to be used by fluent.handler.FluentHandler.
    (included in fluent-logger-python)

    .. versionadded:: 3.17
    """

    def __init__(self, fmt=None, datefmt=None, style='%s'):
        self.datefmt = datefmt
        try:
            self.hostname = socket.gethostname()
        except socket.error:
            self.hostname = None
        self.cmdline = ' '.join(sys.argv)
        try:
            import uwsgi
            svc_name = uwsgi.opt.get('name')
            self.uwsgi_name = svc_name
        except Exception:
            self.uwsgi_name = None

    def formatException(self, exc_info, strip_newlines=True):
        try:
            lines = traceback.format_exception(*exc_info)
        except TypeError as type_error:
            msg = str(type_error)
            lines = ['<Unprintable exception due to %s>\n' % msg]
        if strip_newlines:
            lines = functools.reduce(lambda a, line: a + line.rstrip().splitlines(), lines, [])
        return lines

    def format(self, record):
        message = {'message': record.getMessage(), 'time': self.formatTime(record, self.datefmt), 'name': record.name, 'level': record.levelname, 'filename': record.filename, 'lineno': record.lineno, 'module': record.module, 'funcname': record.funcName, 'process_name': record.processName, 'cmdline': self.cmdline, 'hostname': self.hostname, 'traceback': None, 'error_summary': _get_error_summary(record)}
        context = _update_record_with_context(record)
        if hasattr(record, 'extra'):
            extra = record.extra.copy()
        else:
            extra = {}
        for key in getattr(record, 'extra_keys', []):
            if key not in extra:
                extra[key] = getattr(record, key)
        if 'context' in extra and extra['context']:
            message['context'] = _dictify_context(extra['context'])
        elif context:
            message['context'] = _dictify_context(context)
        else:
            message['context'] = {}
        extra.pop('context', None)
        primitive_types = (str, int, bool, type(None), float, list, dict)
        for key, value in extra.items():
            if not isinstance(value, primitive_types):
                extra[key] = _json_dumps_with_fallback(value)
        message['extra'] = extra
        if record.exc_info:
            message['traceback'] = self.formatException(record.exc_info)
        if self.uwsgi_name:
            message['uwsgi_name'] = self.uwsgi_name
        return message