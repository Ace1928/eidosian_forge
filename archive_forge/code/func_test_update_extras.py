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
def test_update_extras(self):
    a = log.KeywordArgumentAdapter(self.mock_log, {})
    data = {'context': 'some context object', 'instance': 'instance identifier', 'resource_uuid': 'UUID for instance', 'anything': 'goes'}
    expected = copy.copy(data)
    msg, kwargs = a.process('message', data)
    self.assertEqual({'extra': {'anything': expected['anything'], 'context': expected['context'], 'extra_keys': sorted(expected.keys()), 'instance': expected['instance'], 'resource_uuid': expected['resource_uuid']}}, kwargs)