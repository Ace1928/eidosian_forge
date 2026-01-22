import sys
import os
import json
import traceback
import warnings
from io import StringIO, BytesIO
import webob
from webob.exc import HTTPNotFound
from webtest import TestApp
from pecan import (
from pecan.templating import (
from pecan.decorators import accept_noncanonical
from pecan.tests import PecanTestCase
import unittest
class TestDeprecatedRouteMethod(PecanTestCase):

    @property
    def app_(self):

        class RootController(object):

            @expose()
            def index(self, *args):
                return ', '.join(args)

            @expose()
            def _route(self, args):
                return (self.index, args)
        return TestApp(Pecan(RootController()))

    def test_required_argument(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = self.app_.get('/foo/bar/')
            assert r.status_int == 200
            assert b'foo, bar' in r.body