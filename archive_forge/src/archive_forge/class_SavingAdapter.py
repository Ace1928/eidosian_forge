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
class SavingAdapter(log.KeywordArgumentAdapter):

    def __init__(self, *args, **kwds):
        super(log.KeywordArgumentAdapter, self).__init__(*args, **kwds)
        self.results = []

    def process(self, msg, kwargs):
        results = super(SavingAdapter, self).process(msg, kwargs)
        self.results.append((msg, kwargs, results))
        return results