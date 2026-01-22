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
class GeneratorProxy(wrapt.ObjectProxy):

    def __init__(self, wrapped):
        super(GeneratorProxy, self).__init__(wrapped)
        self._self_wrapped = wrapped
        self._self_request_ids = []

    def _set_request_ids(self, resp):
        if self._self_request_ids == []:
            req_id = _extract_request_id(resp)
            self._self_request_ids = [req_id]

    def _next(self):
        obj, resp = next(self._self_wrapped)
        self._set_request_ids(resp)
        return obj

    def next(self):
        return self._next()

    def __next__(self):
        return self._next()

    def __iter__(self):
        return self

    @property
    def request_ids(self):
        return self._self_request_ids

    @property
    def wrapped(self):
        return self._self_wrapped