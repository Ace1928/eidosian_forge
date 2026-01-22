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
class SignedValueTest(unittest.TestCase):
    SECRET = "It's a secret to everybody"
    SECRET_DICT = {0: 'asdfbasdf', 1: '12312312', 2: '2342342'}

    def past(self):
        return self.present() - 86400 * 32

    def present(self):
        return 1300000000

    def test_known_values(self):
        signed_v1 = create_signed_value(SignedValueTest.SECRET, 'key', 'value', version=1, clock=self.present)
        self.assertEqual(signed_v1, b'dmFsdWU=|1300000000|31c934969f53e48164c50768b40cbd7e2daaaa4f')
        signed_v2 = create_signed_value(SignedValueTest.SECRET, 'key', 'value', version=2, clock=self.present)
        self.assertEqual(signed_v2, b'2|1:0|10:1300000000|3:key|8:dmFsdWU=|3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152')
        signed_default = create_signed_value(SignedValueTest.SECRET, 'key', 'value', clock=self.present)
        self.assertEqual(signed_default, signed_v2)
        decoded_v1 = decode_signed_value(SignedValueTest.SECRET, 'key', signed_v1, min_version=1, clock=self.present)
        self.assertEqual(decoded_v1, b'value')
        decoded_v2 = decode_signed_value(SignedValueTest.SECRET, 'key', signed_v2, min_version=2, clock=self.present)
        self.assertEqual(decoded_v2, b'value')

    def test_name_swap(self):
        signed1 = create_signed_value(SignedValueTest.SECRET, 'key1', 'value', clock=self.present)
        signed2 = create_signed_value(SignedValueTest.SECRET, 'key2', 'value', clock=self.present)
        decoded1 = decode_signed_value(SignedValueTest.SECRET, 'key2', signed1, clock=self.present)
        self.assertIs(decoded1, None)
        decoded2 = decode_signed_value(SignedValueTest.SECRET, 'key1', signed2, clock=self.present)
        self.assertIs(decoded2, None)

    def test_expired(self):
        signed = create_signed_value(SignedValueTest.SECRET, 'key1', 'value', clock=self.past)
        decoded_past = decode_signed_value(SignedValueTest.SECRET, 'key1', signed, clock=self.past)
        self.assertEqual(decoded_past, b'value')
        decoded_present = decode_signed_value(SignedValueTest.SECRET, 'key1', signed, clock=self.present)
        self.assertIs(decoded_present, None)

    def test_payload_tampering(self):
        sig = '3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152'

        def validate(prefix):
            return b'value' == decode_signed_value(SignedValueTest.SECRET, 'key', prefix + sig, clock=self.present)
        self.assertTrue(validate('2|1:0|10:1300000000|3:key|8:dmFsdWU=|'))
        self.assertFalse(validate('2|1:1|10:1300000000|3:key|8:dmFsdWU=|'))
        self.assertFalse(validate('2|1:0|10:130000000|3:key|8:dmFsdWU=|'))
        self.assertFalse(validate('2|1:0|10:1300000000|3:keey|8:dmFsdWU=|'))

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

    def test_non_ascii(self):
        value = b'\xe9'
        signed = create_signed_value(SignedValueTest.SECRET, 'key', value, clock=self.present)
        decoded = decode_signed_value(SignedValueTest.SECRET, 'key', signed, clock=self.present)
        self.assertEqual(value, decoded)

    def test_key_versioning_read_write_default_key(self):
        value = b'\xe9'
        signed = create_signed_value(SignedValueTest.SECRET_DICT, 'key', value, clock=self.present, key_version=0)
        decoded = decode_signed_value(SignedValueTest.SECRET_DICT, 'key', signed, clock=self.present)
        self.assertEqual(value, decoded)

    def test_key_versioning_read_write_non_default_key(self):
        value = b'\xe9'
        signed = create_signed_value(SignedValueTest.SECRET_DICT, 'key', value, clock=self.present, key_version=1)
        decoded = decode_signed_value(SignedValueTest.SECRET_DICT, 'key', signed, clock=self.present)
        self.assertEqual(value, decoded)

    def test_key_versioning_invalid_key(self):
        value = b'\xe9'
        signed = create_signed_value(SignedValueTest.SECRET_DICT, 'key', value, clock=self.present, key_version=0)
        newkeys = SignedValueTest.SECRET_DICT.copy()
        newkeys.pop(0)
        decoded = decode_signed_value(newkeys, 'key', signed, clock=self.present)
        self.assertEqual(None, decoded)

    def test_key_version_retrieval(self):
        value = b'\xe9'
        signed = create_signed_value(SignedValueTest.SECRET_DICT, 'key', value, clock=self.present, key_version=1)
        key_version = get_signature_key_version(signed)
        self.assertEqual(1, key_version)