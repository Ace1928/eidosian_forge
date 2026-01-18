from tornado.concurrent import Future
from tornado import gen
from tornado.escape import (
from tornado.httpclient import HTTPClientError
from tornado.httputil import format_timestamp
from tornado.iostream import IOStream
from tornado import locale
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import ignore_deprecation
from tornado.util import ObjectDict, unicode_type
from tornado.web import (
import binascii
import contextlib
import copy
import datetime
import email.utils
import gzip
from io import BytesIO
import itertools
import logging
import os
import re
import socket
import typing  # noqa: F401
import unittest
import urllib.parse
def test_signature_tampering(self):
    prefix = '2|1:0|10:1300000000|3:key|8:dmFsdWU=|'

    def validate(sig):
        return b'value' == decode_signed_value(SignedValueTest.SECRET, 'key', prefix + sig, clock=self.present)
    self.assertTrue(validate('3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152'))
    self.assertFalse(validate('0' * 32))
    self.assertFalse(validate('4d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152'))
    self.assertFalse(validate('3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e153'))
    self.assertFalse(validate('3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e15'))
    self.assertFalse(validate('3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e1538'))