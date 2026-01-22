import eventlet  # noqa
import functools   # noqa: E402
import inspect   # noqa: E402
import os   # noqa: E402
from unittest import mock   # noqa: E402
import fixtures   # noqa: E402
from oslo_concurrency import lockutils   # noqa: E402
from oslo_config import cfg   # noqa: E402
from oslo_config import fixture as config_fixture   # noqa: E402
from oslo_log.fixture import logging_error   # noqa: E402
import testtools   # noqa: E402
from oslo_versionedobjects.tests import obj_fixtures   # noqa: E402
class BaseHookTestCase(TestCase):

    def assert_has_hook(self, expected_name, func):
        self.assertTrue(hasattr(func, '__hook_name__'))
        self.assertEqual(expected_name, func.__hook_name__)