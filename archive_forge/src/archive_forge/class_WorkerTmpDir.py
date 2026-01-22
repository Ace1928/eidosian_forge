import argparse
import copy
import grp
import inspect
import os
import pwd
import re
import shlex
import ssl
import sys
import textwrap
from gunicorn import __version__, util
from gunicorn.errors import ConfigError
from gunicorn.reloader import reloader_engines
class WorkerTmpDir(Setting):
    name = 'worker_tmp_dir'
    section = 'Server Mechanics'
    cli = ['--worker-tmp-dir']
    meta = 'DIR'
    validator = validate_string
    default = None
    desc = '        A directory to use for the worker heartbeat temporary file.\n\n        If not set, the default temporary directory will be used.\n\n        .. note::\n           The current heartbeat system involves calling ``os.fchmod`` on\n           temporary file handlers and may block a worker for arbitrary time\n           if the directory is on a disk-backed filesystem.\n\n           See :ref:`blocking-os-fchmod` for more detailed information\n           and a solution for avoiding this problem.\n        '