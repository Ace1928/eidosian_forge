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
@expose()
def variable_all(self, req, resp, *args, **kwargs):
    data = ['%s=%s' % (key, kwargs[key]) for key in sorted(kwargs.keys())]
    return 'variable_all: %s' % ', '.join(list(args) + data)