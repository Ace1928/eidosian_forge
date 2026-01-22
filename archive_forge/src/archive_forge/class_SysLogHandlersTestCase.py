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
class SysLogHandlersTestCase(BaseTestCase):
    """Test the standard Syslog handler."""

    def setUp(self):
        super(SysLogHandlersTestCase, self).setUp()
        self.facility = logging.handlers.SysLogHandler.LOG_USER
        self.logger = logging.handlers.SysLogHandler(facility=self.facility)

    def test_standard_format(self):
        """Ensure syslog msg isn't modified for standard handler."""
        logrecord = logging.LogRecord('name', logging.WARNING, '/tmp', 1, 'Message', None, None)
        expected = logrecord
        self.assertEqual(expected.getMessage(), self.logger.format(logrecord))