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
def _fix_mime_types(self):
    """Fix incorrect entries in the `mimetypes` registry.

        On Windows, the Python standard library's `mimetypes` reads in
        mappings from file extension to MIME type from the Windows
        registry. Other applications can and do write incorrect values
        to this registry, which causes `mimetypes.guess_type` to return
        incorrect values, which causes TensorBoard to fail to render on
        the frontend.

        This method hard-codes the correct mappings for certain MIME
        types that are known to be either used by TensorBoard or
        problematic in general.
        """
    mimetypes.add_type('text/javascript', '.js')
    mimetypes.add_type('font/woff2', '.woff2')
    mimetypes.add_type('text/html', '.html')