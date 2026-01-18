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
def test_pass_args_to_log(self):
    a = SavingAdapter(self.mock_log, {})
    message = 'message'
    exc_message = 'exception'
    val = 'value'
    a.log(logging.DEBUG, message, name=val, exc_info=exc_message)
    expected = {'exc_info': exc_message, 'extra': {'name': val, 'extra_keys': ['name']}}
    actual = a.results[0]
    self.assertEqual(message, actual[0])
    self.assertEqual(expected, actual[1])
    results = actual[2]
    self.assertEqual(message, results[0])
    self.assertEqual(expected, results[1])