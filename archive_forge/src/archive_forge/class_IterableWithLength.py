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
class IterableWithLength(object):

    def __init__(self, iterable, length):
        self.iterable = iterable
        self.length = length

    def __iter__(self):
        try:
            for chunk in self.iterable:
                yield chunk
        finally:
            self.iterable.close()

    def next(self):
        return next(self.iterable)
    __next__ = next

    def __len__(self):
        return self.length