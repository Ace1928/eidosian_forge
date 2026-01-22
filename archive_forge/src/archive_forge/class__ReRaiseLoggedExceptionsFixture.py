import datetime
import logging
import sys
import uuid
import fixtures
from kombu import connection
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import timeutils
from stevedore import dispatch
from stevedore import extension
import testscenarios
import yaml
import oslo_messaging
from oslo_messaging.notify import _impl_log
from oslo_messaging.notify import _impl_test
from oslo_messaging.notify import messaging
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class _ReRaiseLoggedExceptionsFixture(fixtures.Fixture):
    """Record logged exceptions and re-raise in cleanup.

    The notifier just logs notification send errors so, for the sake of
    debugging test failures, we record any exceptions logged and re-raise them
    during cleanup.
    """

    class FakeLogger(object):

        def __init__(self):
            self.exceptions = []

        def exception(self, msg, *args, **kwargs):
            self.exceptions.append(sys.exc_info()[1])

        def warning(self, msg, *args, **kwargs):
            return

    def setUp(self):
        super(_ReRaiseLoggedExceptionsFixture, self).setUp()
        self.logger = self.FakeLogger()

        def reraise_exceptions():
            for ex in self.logger.exceptions:
                raise ex
        self.addCleanup(reraise_exceptions)