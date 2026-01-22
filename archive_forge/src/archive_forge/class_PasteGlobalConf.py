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
class PasteGlobalConf(Setting):
    name = 'raw_paste_global_conf'
    action = 'append'
    section = 'Server Mechanics'
    cli = ['--paste-global']
    meta = 'CONF'
    validator = validate_list_string
    default = []
    desc = '        Set a PasteDeploy global config variable in ``key=value`` form.\n\n        The option can be specified multiple times.\n\n        The variables are passed to the the PasteDeploy entrypoint. Example::\n\n            $ gunicorn -b 127.0.0.1:8000 --paste development.ini --paste-global FOO=1 --paste-global BAR=2\n\n        .. versionadded:: 19.7\n        '