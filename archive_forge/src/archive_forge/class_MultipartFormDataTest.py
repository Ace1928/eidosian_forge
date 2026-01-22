from tornado.httputil import (
from tornado.escape import utf8, native_str
from tornado.log import gen_log
from tornado.testing import ExpectLog
from tornado.test.util import ignore_deprecation
import copy
import datetime
import logging
import pickle
import time
import urllib.parse
import unittest
from typing import Tuple, Dict, List
class MultipartFormDataTest(unittest.TestCase):

    def test_file_upload(self):
        data = b'--1234\nContent-Disposition: form-data; name="files"; filename="ab.txt"\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        args, files = form_data_args()
        parse_multipart_form_data(b'1234', data, args, files)
        file = files['files'][0]
        self.assertEqual(file['filename'], 'ab.txt')
        self.assertEqual(file['body'], b'Foo')

    def test_unquoted_names(self):
        data = b'--1234\nContent-Disposition: form-data; name=files; filename=ab.txt\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        args, files = form_data_args()
        parse_multipart_form_data(b'1234', data, args, files)
        file = files['files'][0]
        self.assertEqual(file['filename'], 'ab.txt')
        self.assertEqual(file['body'], b'Foo')

    def test_special_filenames(self):
        filenames = ['a;b.txt', 'a"b.txt', 'a";b.txt', 'a;"b.txt', 'a";";.txt', 'a\\"b.txt', 'a\\b.txt']
        for filename in filenames:
            logging.debug('trying filename %r', filename)
            str_data = '--1234\nContent-Disposition: form-data; name="files"; filename="%s"\n\nFoo\n--1234--' % filename.replace('\\', '\\\\').replace('"', '\\"')
            data = utf8(str_data.replace('\n', '\r\n'))
            args, files = form_data_args()
            parse_multipart_form_data(b'1234', data, args, files)
            file = files['files'][0]
            self.assertEqual(file['filename'], filename)
            self.assertEqual(file['body'], b'Foo')

    def test_non_ascii_filename(self):
        data = b'--1234\nContent-Disposition: form-data; name="files"; filename="ab.txt"; filename*=UTF-8\'\'%C3%A1b.txt\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        args, files = form_data_args()
        parse_multipart_form_data(b'1234', data, args, files)
        file = files['files'][0]
        self.assertEqual(file['filename'], 'áb.txt')
        self.assertEqual(file['body'], b'Foo')

    def test_boundary_starts_and_ends_with_quotes(self):
        data = b'--1234\nContent-Disposition: form-data; name="files"; filename="ab.txt"\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        args, files = form_data_args()
        parse_multipart_form_data(b'"1234"', data, args, files)
        file = files['files'][0]
        self.assertEqual(file['filename'], 'ab.txt')
        self.assertEqual(file['body'], b'Foo')

    def test_missing_headers(self):
        data = b'--1234\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        args, files = form_data_args()
        with ExpectLog(gen_log, 'multipart/form-data missing headers'):
            parse_multipart_form_data(b'1234', data, args, files)
        self.assertEqual(files, {})

    def test_invalid_content_disposition(self):
        data = b'--1234\nContent-Disposition: invalid; name="files"; filename="ab.txt"\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        args, files = form_data_args()
        with ExpectLog(gen_log, 'Invalid multipart/form-data'):
            parse_multipart_form_data(b'1234', data, args, files)
        self.assertEqual(files, {})

    def test_line_does_not_end_with_correct_line_break(self):
        data = b'--1234\nContent-Disposition: form-data; name="files"; filename="ab.txt"\n\nFoo--1234--'.replace(b'\n', b'\r\n')
        args, files = form_data_args()
        with ExpectLog(gen_log, 'Invalid multipart/form-data'):
            parse_multipart_form_data(b'1234', data, args, files)
        self.assertEqual(files, {})

    def test_content_disposition_header_without_name_parameter(self):
        data = b'--1234\nContent-Disposition: form-data; filename="ab.txt"\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        args, files = form_data_args()
        with ExpectLog(gen_log, 'multipart/form-data value missing name'):
            parse_multipart_form_data(b'1234', data, args, files)
        self.assertEqual(files, {})

    def test_data_after_final_boundary(self):
        data = b'--1234\nContent-Disposition: form-data; name="files"; filename="ab.txt"\n\nFoo\n--1234--\n'.replace(b'\n', b'\r\n')
        args, files = form_data_args()
        parse_multipart_form_data(b'1234', data, args, files)
        file = files['files'][0]
        self.assertEqual(file['filename'], 'ab.txt')
        self.assertEqual(file['body'], b'Foo')