import datetime
import io
import os
import re
import signal
import sys
import threading
from unittest import mock
import fixtures
import greenlet
from oslotest import base
import oslo_config
from oslo_config import fixture
from oslo_reports import guru_meditation_report as gmr
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import opts
class GmrConfigFixture(fixture.Config):

    def setUp(self):
        super(GmrConfigFixture, self).setUp()
        self.conf.set_override('file_event_handler', '/specific/file', group='oslo_reports')
        self.conf.set_override('file_event_handler_interval', 10, group='oslo_reports')
        self.conf.set_override('log_dir', '/var/fake_log', group='oslo_reports')