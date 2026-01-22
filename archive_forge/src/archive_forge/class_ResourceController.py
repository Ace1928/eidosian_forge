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
class ResourceController(object):

    def __init__(self, name):
        self.name = name
        assert self.name == 'file.html'

    @expose('json')
    def index(self):
        return dict(name=self.name)