import time
from json import dumps, loads
import warnings
from unittest import mock
from webtest import TestApp
import webob
from pecan import Pecan, expose, abort, Request, Response
from pecan.rest import RestController
from pecan.hooks import PecanHook, HookController
from pecan.tests import PecanTestCase
def test_other_custom_action_with_method_parameter(self):
    r = self.app_.post('/things/other', {'_method': 'MISC'}, status=405)
    assert r.status_int == 405