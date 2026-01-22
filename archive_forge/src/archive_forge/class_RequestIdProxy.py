import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
class RequestIdProxy(wrapt.ObjectProxy):

    def __init__(self, wrapped):
        super(RequestIdProxy, self).__init__(wrapped[0])
        self._self_wrapped = wrapped[0]
        req_id = _extract_request_id(wrapped[1])
        self._self_request_ids = [req_id]

    @property
    def request_ids(self):
        return self._self_request_ids

    @property
    def wrapped(self):
        return self._self_wrapped

    def next(self):
        return next(self._self_wrapped)
    __next__ = next