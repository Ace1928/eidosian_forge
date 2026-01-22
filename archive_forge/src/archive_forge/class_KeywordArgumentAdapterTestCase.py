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
class KeywordArgumentAdapterTestCase(BaseTestCase):

    def setUp(self):
        super(KeywordArgumentAdapterTestCase, self).setUp()
        self.mock_log = mock.Mock()
        self.mock_log.manager.disable = logging.NOTSET
        self.mock_log.isEnabledFor.return_value = True
        self.mock_log.getEffectiveLevel.return_value = logging.DEBUG

    def test_empty_kwargs(self):
        a = log.KeywordArgumentAdapter(self.mock_log, {})
        msg, kwargs = a.process('message', {})
        self.assertEqual({'extra': {'extra_keys': []}}, kwargs)

    def test_include_constructor_extras(self):
        key = 'foo'
        val = 'blah'
        data = {key: val}
        a = log.KeywordArgumentAdapter(self.mock_log, data)
        msg, kwargs = a.process('message', {})
        self.assertEqual({'extra': {key: val, 'extra_keys': [key]}}, kwargs)

    def test_pass_through_exc_info(self):
        a = log.KeywordArgumentAdapter(self.mock_log, {})
        exc_message = 'exception'
        msg, kwargs = a.process('message', {'exc_info': exc_message})
        self.assertEqual({'extra': {'extra_keys': []}, 'exc_info': exc_message}, kwargs)

    def test_update_extras(self):
        a = log.KeywordArgumentAdapter(self.mock_log, {})
        data = {'context': 'some context object', 'instance': 'instance identifier', 'resource_uuid': 'UUID for instance', 'anything': 'goes'}
        expected = copy.copy(data)
        msg, kwargs = a.process('message', data)
        self.assertEqual({'extra': {'anything': expected['anything'], 'context': expected['context'], 'extra_keys': sorted(expected.keys()), 'instance': expected['instance'], 'resource_uuid': expected['resource_uuid']}}, kwargs)

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

    def test_pass_args_via_debug(self):
        a = SavingAdapter(self.mock_log, {})
        message = 'message'
        exc_message = 'exception'
        val = 'value'
        a.debug(message, name=val, exc_info=exc_message)
        expected = {'exc_info': exc_message, 'extra': {'name': val, 'extra_keys': ['name']}}
        actual = a.results[0]
        self.assertEqual(message, actual[0])
        self.assertEqual(expected, actual[1])
        results = actual[2]
        self.assertEqual(message, results[0])
        self.assertEqual(expected, results[1])