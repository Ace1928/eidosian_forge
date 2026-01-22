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
def _should_use_data_server(flags):
    if flags.logdir_spec and (not flags.logdir):
        logger.info('Note: --logdir_spec is not supported with --load_fast behavior; falling back to slower Python-only load path. To use the data server, replace --logdir_spec with --logdir.')
        return False
    if not flags.logdir:
        return False
    if '://' in flags.logdir and (not flags.logdir.startswith('gs://')):
        logger.info('Note: --load_fast behavior only supports local and GCS (gs://) paths; falling back to slower Python-only load path.')
        return False
    if flags.detect_file_replacement is True:
        logger.info('Note: --detect_file_replacement=true is not supported with --load_fast behavior; falling back to slower Python-only load path.')
        return False
    return True