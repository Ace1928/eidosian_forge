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
class CustomRequest(Request):

    @property
    def headers(self):
        headers = super(CustomRequest, self).headers
        headers['X-Custom-Request'] = 'ABC'
        return headers