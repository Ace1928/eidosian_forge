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
class TestScriptName(PecanTestCase):

    def setUp(self):
        super(TestScriptName, self).setUp()
        self.environ = {'SCRIPT_NAME': '/foo'}

    def test_handle_script_name(self):

        class RootController(object):

            @expose()
            def index(self):
                return 'Root Index'
        app = TestApp(Pecan(RootController()), extra_environ=self.environ)
        r = app.get('/foo/')
        assert r.status_int == 200