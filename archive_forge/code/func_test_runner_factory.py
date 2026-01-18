from functools import reduce
import gc
import io
import locale  # system locale module, not tornado.locale
import logging
import operator
import textwrap
import sys
import unittest
import warnings
from tornado.httpclient import AsyncHTTPClient
from tornado.httpserver import HTTPServer
from tornado.netutil import Resolver
from tornado.options import define, add_parse_callback, options
def test_runner_factory(stderr):

    class TornadoTextTestRunner(unittest.TextTestRunner):

        def __init__(self, *args, **kwargs):
            kwargs['stream'] = stderr
            super().__init__(*args, **kwargs)

        def run(self, test):
            result = super().run(test)
            if result.skipped:
                skip_reasons = set((reason for test, reason in result.skipped))
                self.stream.write(textwrap.fill('Some tests were skipped because: %s' % ', '.join(sorted(skip_reasons))))
                self.stream.write('\n')
            return result
    return TornadoTextTestRunner