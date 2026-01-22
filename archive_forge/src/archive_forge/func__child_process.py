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
def _child_process(self, service):
    self._child_process_handle_signal()
    eventlet.hubs.use_hub()
    os.close(self.writepipe)
    eventlet.spawn_n(self._pipe_watcher)
    random.seed()
    launcher = Launcher(self.conf, restart_method=self.restart_method)
    launcher.launch_service(service)
    return launcher