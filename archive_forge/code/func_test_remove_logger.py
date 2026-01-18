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
def test_remove_logger(self):
    fake_handler = {'class': 'logging.StreamHandler'}
    fake_logger = {'level': 'WARN'}
    conf1 = {'root': {'handlers': 'fake'}, 'handlers': {'fake': fake_handler}, 'loggers': {'a.a': fake_logger}}
    conf2 = {'root': {'handlers': 'fake'}, 'handlers': {'fake': fake_handler}}
    stream = io.StringIO()
    with self.mutate_conf(conf1, conf2) as (loginis, confs):
        stream = self.set_root_stream()
        log = logging.getLogger('a.a')
        log.info('info')
        log.warn('warn')
        self.assertEqual('warn\n', stream.getvalue())
    stream = self.set_root_stream()
    log.info('info')
    log.warn('warn')
    self.assertEqual('info\nwarn\n', stream.getvalue())