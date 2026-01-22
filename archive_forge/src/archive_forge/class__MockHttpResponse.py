import socket
import unittest
import httplib2
from six.moves import http_client
from mock import patch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
class _MockHttpResponse(object):

    def __init__(self, status_code):
        self.response = {'status': status_code}