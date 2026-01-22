import abc
import collections
import copy
import errno
import functools
import gc
import inspect
import io
import logging
import os
import random
import signal
import sys
import time
import eventlet
from eventlet import event
from eventlet import tpool
from oslo_concurrency import lockutils
from oslo_service._i18n import _
from oslo_service import _options
from oslo_service import eventlet_backdoor
from oslo_service import systemd
from oslo_service import threadgroup
def _child_wait_for_exit_or_signal(self, launcher):
    status = 0
    signo = 0
    try:
        launcher.wait()
    except SignalExit as exc:
        signame = self.signal_handler.signals_to_name[exc.signo]
        LOG.info('Child caught %s, handling', signame)
        status = exc.code
        signo = exc.signo
    except SystemExit as exc:
        launcher.stop()
        status = exc.code
    except BaseException:
        launcher.stop()
        LOG.exception('Unhandled exception')
        status = 2
    return (status, signo)