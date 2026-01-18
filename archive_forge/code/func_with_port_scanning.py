from abc import ABCMeta
from abc import abstractmethod
import argparse
import atexit
from collections import defaultdict
import errno
import logging
import mimetypes
import os
import shlex
import signal
import socket
import sys
import threading
import time
import urllib.parse
from absl import flags as absl_flags
from absl.flags import argparse_flags
from werkzeug import serving
from tensorboard import manager
from tensorboard import version
from tensorboard.backend import application
from tensorboard.backend.event_processing import data_ingester as local_ingester
from tensorboard.backend.event_processing import event_file_inspector as efi
from tensorboard.data import server_ingester
from tensorboard.plugins.core import core_plugin
from tensorboard.util import tb_logging
def with_port_scanning(cls):
    """Create a server factory that performs port scanning.

    This function returns a callable whose signature matches the
    specification of `TensorBoardServer.__init__`, using `cls` as an
    underlying implementation. It passes through `flags` unchanged except
    in the case that `flags.port is None`, in which case it repeatedly
    instantiates the underlying server with new port suggestions.

    Args:
      cls: A valid implementation of `TensorBoardServer`. This class's
        initializer should raise a `TensorBoardPortInUseError` upon
        failing to bind to a port when it is expected that binding to
        another nearby port might succeed.

        The initializer for `cls` will only ever be invoked with `flags`
        such that `flags.port is not None`.

    Returns:
      A function that implements the `__init__` contract of
      `TensorBoardServer`.
    """

    def init(wsgi_app, flags):
        should_scan = flags.port is None
        base_port = core_plugin.DEFAULT_PORT if flags.port is None else flags.port
        if base_port > 65535:
            raise TensorBoardServerException('TensorBoard cannot bind to port %d > %d' % (base_port, 65535))
        max_attempts = 100 if should_scan else 1
        base_port = min(base_port + max_attempts, 65536) - max_attempts
        for port in range(base_port, base_port + max_attempts):
            subflags = argparse.Namespace(**vars(flags))
            subflags.port = port
            try:
                return cls(wsgi_app=wsgi_app, flags=subflags)
            except TensorBoardPortInUseError:
                if not should_scan:
                    raise
        raise TensorBoardServerException('TensorBoard could not bind to any port around %s (tried %d times)' % (base_port, max_attempts))
    return init