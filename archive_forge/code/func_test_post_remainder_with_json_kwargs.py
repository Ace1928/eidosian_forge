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
def test_post_remainder_with_json_kwargs(self):
    r = self.app_.post_json('/eater/9', {'month': '1', 'day': '12'})
    assert r.status_int == 200
    assert r.body == b'eater: 9, None, day=12, month=1'