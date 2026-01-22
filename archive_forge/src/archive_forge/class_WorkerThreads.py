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
class WorkerThreads(Setting):
    name = 'threads'
    section = 'Worker Processes'
    cli = ['--threads']
    meta = 'INT'
    validator = validate_pos_int
    type = int
    default = 1
    desc = "        The number of worker threads for handling requests.\n\n        Run each worker with the specified number of threads.\n\n        A positive integer generally in the ``2-4 x $(NUM_CORES)`` range.\n        You'll want to vary this a bit to find the best for your particular\n        application's work load.\n\n        If it is not defined, the default is ``1``.\n\n        This setting only affects the Gthread worker type.\n\n        .. note::\n           If you try to use the ``sync`` worker type and set the ``threads``\n           setting to more than 1, the ``gthread`` worker type will be used\n           instead.\n        "