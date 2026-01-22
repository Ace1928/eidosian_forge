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
def _install_signal_handler(self, signal_number, signal_name):
    """Set a signal handler to gracefully exit on the given signal.

        When this process receives the given signal, it will run `atexit`
        handlers and then exit with `0`.

        Args:
          signal_number: The numeric code for the signal to handle, like
            `signal.SIGTERM`.
          signal_name: The human-readable signal name.
        """
    old_signal_handler = None

    def handler(handled_signal_number, frame):
        signal.signal(signal_number, signal.SIG_DFL)
        sys.stderr.write('TensorBoard caught %s; exiting...\n' % signal_name)
        if old_signal_handler not in (signal.SIG_IGN, signal.SIG_DFL):
            old_signal_handler(handled_signal_number, frame)
        sys.exit(0)
    old_signal_handler = signal.signal(signal_number, handler)