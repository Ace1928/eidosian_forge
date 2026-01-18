import io
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _message
from oslo_i18n import log as i18n_log
from oslo_i18n.tests import fakes
def test_set_formatter(self):
    formatter = 'some formatter'
    self.translation_handler.setFormatter(formatter)
    self.assertEqual(formatter, self.translation_handler.target.formatter)