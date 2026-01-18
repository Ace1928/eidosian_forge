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
def test_user_identity_logging_set_format(self):
    self.config(logging_context_format_string='HAS CONTEXT [%(request_id)s %(user_identity)s]: %(message)s', logging_user_identity_format='%(user)s %(project)s')
    ctxt = _fake_context()
    ctxt.request_id = '99'
    message = 'test'
    self.log.info(message, context=ctxt)
    expected = 'HAS CONTEXT [%s %s %s]: %s\n' % (ctxt.request_id, ctxt.user, ctxt.project_id, str(message))
    self.assertEqual(expected, self.stream.getvalue())