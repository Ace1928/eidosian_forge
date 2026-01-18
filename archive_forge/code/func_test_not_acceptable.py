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
def test_not_acceptable(self):
    r = self.app_.get('/', headers={'Accept': 'application/xml'}, status=406)
    assert r.status_int == 406