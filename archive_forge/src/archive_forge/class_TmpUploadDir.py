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
class TmpUploadDir(Setting):
    name = 'tmp_upload_dir'
    section = 'Server Mechanics'
    meta = 'DIR'
    validator = validate_string
    default = None
    desc = '        Directory to store temporary request data as they are read.\n\n        This may disappear in the near future.\n\n        This path should be writable by the process permissions set for Gunicorn\n        workers. If not specified, Gunicorn will choose a system generated\n        temporary directory.\n        '