import logging
import sys
from unittest import mock
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
from oslotest import base as test_base
from oslo_log import formatters
from oslo_log import log
def test_replace_false_value_not_exists(self):
    d = {'user': 'user1'}
    s = '%(project)s' % formatters._ReplaceFalseValue(d)
    self.assertEqual('-', s)