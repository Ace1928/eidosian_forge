import inspect
import logging
import os
import signal
import stat
import sys
import threading
import time
import traceback
from oslo_utils import timeutils
from oslo_reports.generators import conf as cgen
from oslo_reports.generators import process as prgen
from oslo_reports.generators import threading as tgen
from oslo_reports.generators import version as pgen
from oslo_reports import report
@classmethod
def _setup_signal(cls, signum, version, service_name, log_dir):
    signal.signal(signum, lambda sn, f: cls.handle_signal(version, service_name, log_dir, f))