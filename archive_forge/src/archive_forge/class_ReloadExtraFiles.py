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
class ReloadExtraFiles(Setting):
    name = 'reload_extra_files'
    action = 'append'
    section = 'Debugging'
    cli = ['--reload-extra-file']
    meta = 'FILES'
    validator = validate_list_of_existing_files
    default = []
    desc = '        Extends :ref:`reload` option to also watch and reload on additional files\n        (e.g., templates, configurations, specifications, etc.).\n\n        .. versionadded:: 19.8\n        '