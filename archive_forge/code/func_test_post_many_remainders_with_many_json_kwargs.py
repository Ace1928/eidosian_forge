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
def test_post_many_remainders_with_many_json_kwargs(self):
    r = self.app_.post_json('/eater/10', {'id': 'ten', 'month': '1', 'day': '12', 'dummy': 'dummy'})
    assert r.status_int == 200
    assert r.body == b'eater: 10, dummy, day=12, month=1'