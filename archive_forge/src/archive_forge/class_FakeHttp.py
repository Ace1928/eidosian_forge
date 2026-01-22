import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
class FakeHttp(object):

    class FakeRequest(object):

        def __init__(self, credentials=None):
            if credentials is not None:
                self.credentials = credentials

    def __init__(self, credentials=None):
        self.request = FakeHttp.FakeRequest(credentials=credentials)