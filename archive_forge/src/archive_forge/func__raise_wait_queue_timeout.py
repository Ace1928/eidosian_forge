from __future__ import annotations
import collections
import contextlib
import copy
import os
import platform
import socket
import ssl
import sys
import threading
import time
import weakref
from typing import (
import bson
from bson import DEFAULT_CODEC_OPTIONS
from bson.son import SON
from pymongo import __version__, _csot, auth, helpers
from pymongo.client_session import _validate_session_write_concern
from pymongo.common import (
from pymongo.errors import (
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.lock import _create_lock
from pymongo.monitoring import (
from pymongo.network import command, receive_message
from pymongo.read_preferences import ReadPreference
from pymongo.server_api import _add_to_command
from pymongo.server_type import SERVER_TYPE
from pymongo.socket_checker import SocketChecker
from pymongo.ssl_support import HAS_SNI, SSLError
def _raise_wait_queue_timeout(self) -> NoReturn:
    listeners = self.opts._event_listeners
    if self.enabled_for_cmap:
        assert listeners is not None
        listeners.publish_connection_check_out_failed(self.address, ConnectionCheckOutFailedReason.TIMEOUT)
    timeout = _csot.get_timeout() or self.opts.wait_queue_timeout
    if self.opts.load_balanced:
        other_ops = self.active_sockets - self.ncursors - self.ntxns
        raise WaitQueueTimeoutError('Timeout waiting for connection from the connection pool. maxPoolSize: {}, connections in use by cursors: {}, connections in use by transactions: {}, connections in use by other operations: {}, timeout: {}'.format(self.opts.max_pool_size, self.ncursors, self.ntxns, other_ops, timeout))
    raise WaitQueueTimeoutError(f'Timed out while checking out a connection from connection pool. maxPoolSize: {self.opts.max_pool_size}, timeout: {timeout}')