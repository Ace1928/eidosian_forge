from contextlib import contextmanager
import copy
import datetime
import io
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from unittest import mock
from dateutil import tz
from oslo_config import cfg
from oslo_config import fixture as fixture_config  # noqa
from oslo_context import context
from oslo_context import fixture as fixture_context
from oslo_i18n import fixture as fixture_trans
from oslo_serialization import jsonutils
from oslotest import base as test_base
import testtools
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
from oslo_log import log
from oslo_utils import units
def test_skip_logging_builtin_exceptions(self):
    ctxt = _fake_context()
    ctxt.request_id = str('99')
    message = self.trans_fixture.lazy('test ' + chr(128))
    ignored_exceptions = [ValueError, TypeError, KeyError, AttributeError, ImportError]
    for ignore in ignored_exceptions:
        try:
            raise ignore('test_exception_logging')
        except ignore as e:
            self.log.warning(message, context=ctxt)
            expected = '{}: {}'.format(e.__class__.__name__, e)
        self.assertNotIn(expected, self.stream.getvalue())