import abc
import errno
import functools
import os
import re
import signal
import struct
import subprocess
import sys
import time
from eventlet.green import socket
import eventlet.greenio
import eventlet.wsgi
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_concurrency import processutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
from osprofiler import opts as profiler_opts
import routes.middleware
import webob.dec
import webob.exc
from webob import multidict
from glance.common import config
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
import glance.db
from glance import housekeeping
from glance import i18n
from glance.i18n import _, _LE, _LI, _LW
from glance import sqlite_migration
def _get_sock_from_parent(self):
    pipe_handle = int(getattr(CONF, 'pipe_handle', 0))
    if not pipe_handle:
        err_msg = _('Did not receive a pipe handle, which is used when communicating with the parent process.')
        raise exception.GlanceException(err_msg)
    buff = self._ioutils.get_buffer(4)
    self._ioutils.read_file(pipe_handle, buff, 4)
    socket_buff_sz = struct.unpack('<I', buff)[0]
    socket_buff = self._ioutils.get_buffer(socket_buff_sz)
    self._ioutils.read_file(pipe_handle, socket_buff, socket_buff_sz)
    self._ioutils.close_handle(pipe_handle)
    return socket.fromshare(bytes(socket_buff[:]))