from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from collections import OrderedDict
import contextlib
import copy
import datetime
import json
import logging
import os
import sys
import time
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import parser as style_parser
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def GetErrorDict(self, log_record):
    """Extract exception info from a logging.LogRecord as an OrderedDict."""
    error_dict = OrderedDict()
    if log_record.exc_info:
        if not log_record.exc_text:
            log_record.exc_text = self.formatException(log_record.exc_info)
        if issubclass(type(log_record.msg), BaseException):
            error_dict['type'] = type(log_record.msg).__name__
            error_dict['details'] = six.text_type(log_record.msg)
            error_dict['stacktrace'] = getattr(log_record.msg, '__traceback__', None)
        elif issubclass(type(log_record.exc_info[0]), BaseException):
            error_dict['type'] = log_record.exc_info[0]
            error_dict['details'] = log_record.exc_text
            error_dict['stacktrace'] = log_record.exc_info[2]
        else:
            error_dict['type'] = log_record.exc_text
            error_dict['details'] = log_record.exc_text
            error_dict['stacktrace'] = log_record.exc_text
        return error_dict
    return None