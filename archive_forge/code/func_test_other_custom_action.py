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
def test_other_custom_action(self):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = self.app_.request('/things/other', method='MISC', status=405)
        assert r.status_int == 405