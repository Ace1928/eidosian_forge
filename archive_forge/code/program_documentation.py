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
Fix werkzeug logging setup so it inherits TensorBoard's log level.

        This addresses a change in werkzeug 0.15.0+ [1] that causes it set its own
        log level to INFO regardless of the root logger configuration. We instead
        want werkzeug to inherit TensorBoard's root logger log level (set via absl
        to WARNING by default).

        [1]: https://github.com/pallets/werkzeug/commit/4cf77d25858ff46ac7e9d64ade054bf05b41ce12
        