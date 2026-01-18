import datetime
import decimal
import json
import warnings
from importlib import import_module
import sqlparse
from django.conf import settings
from django.db import NotSupportedError, transaction
from django.db.backends import utils
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.encoding import force_str
def window_frame_start(self, start):
    if isinstance(start, int):
        if start < 0:
            return '%d %s' % (abs(start), self.PRECEDING)
        elif start == 0:
            return self.CURRENT_ROW
    elif start is None:
        return self.UNBOUNDED_PRECEDING
    raise ValueError("start argument must be a negative integer, zero, or None, but got '%s'." % start)