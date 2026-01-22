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
class MaxRequestsJitter(Setting):
    name = 'max_requests_jitter'
    section = 'Worker Processes'
    cli = ['--max-requests-jitter']
    meta = 'INT'
    validator = validate_pos_int
    type = int
    default = 0
    desc = '        The maximum jitter to add to the *max_requests* setting.\n\n        The jitter causes the restart per worker to be randomized by\n        ``randint(0, max_requests_jitter)``. This is intended to stagger worker\n        restarts to avoid all workers restarting at the same time.\n\n        .. versionadded:: 19.2\n        '