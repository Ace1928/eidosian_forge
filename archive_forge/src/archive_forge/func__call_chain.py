import urllib.request
import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings
from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
from urllib.response import addinfourl, addclosehook
def _call_chain(self, chain, kind, meth_name, *args):
    handlers = chain.get(kind, ())
    for handler in handlers:
        func = getattr(handler, meth_name)
        result = func(*args)
        if result is not None:
            return result