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
def _clear_attrs(self):
    for key in [k for k in self.__dict__.keys() if k[0] != '_']:
        del self.__dict__[key]