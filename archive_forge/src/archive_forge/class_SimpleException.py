import sys
from boto.compat import json
from boto.exception import BotoServerError
class SimpleException(BotoServerError):

    def __init__(self, e):
        super(SimpleException, self).__init__(e.status, e.reason, e.body)
        self.error_message = self.message

    def __repr__(self):
        return self.__class__.__name__ + ': ' + self.error_message

    def __str__(self):
        return self.__class__.__name__ + ': ' + self.error_message