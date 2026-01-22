import sys
from oslo_serialization import jsonutils
import testscenarios
import oslo_messaging
from oslo_messaging._drivers import common as exceptions
from oslo_messaging.tests import utils as test_utils
class NovaStyleException(Exception):
    format = 'I am Nova'

    def __init__(self, message=None, **kwargs):
        self.kwargs = kwargs
        if not message:
            message = self.format % kwargs
        super(NovaStyleException, self).__init__(message)