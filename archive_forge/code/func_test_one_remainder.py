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
def test_one_remainder(self):
    r = self.app_.get('/eater/1')
    assert r.status_int == 200
    assert r.body == b'eater: 1, None, '