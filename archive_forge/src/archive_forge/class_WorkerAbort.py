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
class WorkerAbort(Setting):
    name = 'worker_abort'
    section = 'Server Hooks'
    validator = validate_callable(1)
    type = callable

    def worker_abort(worker):
        pass
    default = staticmethod(worker_abort)
    desc = '        Called when a worker received the SIGABRT signal.\n\n        This call generally happens on timeout.\n\n        The callable needs to accept one instance variable for the initialized\n        Worker.\n        '