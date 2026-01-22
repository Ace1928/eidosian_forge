from unittest import mock
from oslo_serialization import jsonutils
import sys
from keystoneauth1 import fixture
import requests
class FakeLog(object):

    def __init__(self):
        self.messages = {}

    def debug(self, msg):
        self.messages['debug'] = msg

    def info(self, msg):
        self.messages['info'] = msg

    def warning(self, msg):
        self.messages['warning'] = msg

    def error(self, msg):
        self.messages['error'] = msg

    def critical(self, msg):
        self.messages['critical'] = msg