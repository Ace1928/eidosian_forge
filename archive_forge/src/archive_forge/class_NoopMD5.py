import base64
from calendar import timegm
from collections.abc import Mapping
import gzip
import hashlib
import hmac
import io
import json
import logging
import time
import traceback
class NoopMD5:

    def __init__(self, *a, **kw):
        pass

    def update(self, *a, **kw):
        pass

    def hexdigest(self, *a, **kw):
        return ''