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
def _start_child(self, wrap):
    if len(wrap.forktimes) > wrap.workers:
        if time.time() - wrap.forktimes[0] < wrap.workers:
            LOG.info('Forking too fast, sleeping')
            time.sleep(1)
        wrap.forktimes.pop(0)
    wrap.forktimes.append(time.time())
    pid = os.fork()
    if pid == 0:
        tpool.killall()
        self.launcher = self._child_process(wrap.service)
        while True:
            self._child_process_handle_signal()
            status, signo = self._child_wait_for_exit_or_signal(self.launcher)
            if not _is_sighup_and_daemon(signo):
                self.launcher.wait()
                break
            self.launcher.restart()
        os._exit(status)
    LOG.debug('Started child %d', pid)
    wrap.children.add(pid)
    self.children[pid] = wrap
    return pid