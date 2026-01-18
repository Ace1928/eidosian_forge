import base64
import datetime
import json
import time
import warnings
import zlib
from django.conf import settings
from django.utils.crypto import constant_time_compare, salted_hmac
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.encoding import force_bytes
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
def unsign(self, value, max_age=None):
    """
        Retrieve original value and check it wasn't signed more
        than max_age seconds ago.
        """
    result = super().unsign(value)
    value, timestamp = result.rsplit(self.sep, 1)
    timestamp = b62_decode(timestamp)
    if max_age is not None:
        if isinstance(max_age, datetime.timedelta):
            max_age = max_age.total_seconds()
        age = time.time() - timestamp
        if age > max_age:
            raise SignatureExpired('Signature age %s > %s seconds' % (age, max_age))
    return value