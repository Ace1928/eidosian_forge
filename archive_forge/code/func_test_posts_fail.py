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
def test_posts_fail(self):
    try:
        self.app_.post('/sub', dict(foo=1))
        raise Exception('Post should fail')
    except Exception as e:
        assert isinstance(e, RuntimeError)