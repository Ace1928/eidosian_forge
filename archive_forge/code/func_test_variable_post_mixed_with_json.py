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
def test_variable_post_mixed_with_json(self):
    r = self.app_.post_json('/variable_all/7', {'id': 'seven', 'month': '1', 'day': '12'})
    assert r.status_int == 200
    assert r.body == b'variable_all: 7, day=12, id=seven, month=1'