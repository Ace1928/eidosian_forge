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
def window_frame_end(self, end):
    if isinstance(end, int):
        if end == 0:
            return self.CURRENT_ROW
        elif end > 0:
            return '%d %s' % (end, self.FOLLOWING)
    elif end is None:
        return self.UNBOUNDED_FOLLOWING
    raise ValueError("end argument must be a positive integer, zero, or None, but got '%s'." % end)