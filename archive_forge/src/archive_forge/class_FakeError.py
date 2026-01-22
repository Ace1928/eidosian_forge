from boto.beanstalk.exception import simple
from tests.compat import unittest
class FakeError(object):

    def __init__(self, code, status, reason, body):
        self.code = code
        self.status = status
        self.reason = reason
        self.body = body