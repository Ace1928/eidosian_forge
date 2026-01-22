import base64
import functools
import inspect
import json
import logging
import os
import warnings
import six
from six.moves import urllib
def _urlsafe_b64decode(b64string):
    b64string = _to_bytes(b64string)
    padded = b64string + b'=' * (4 - len(b64string) % 4)
    return base64.urlsafe_b64decode(padded)