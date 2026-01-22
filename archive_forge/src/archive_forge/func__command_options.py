import inspect
import os
import sys
import breezy
from . import commands as _mod_commands
from . import errors, help_topics, option
from . import plugin as _mod_plugin
from .i18n import gettext
from .trace import mutter, note
def _command_options(exporter, context, cmd):
    note = f'option of {cmd.name()!r} command'
    for opt in cmd.takes_options:
        if not isinstance(opt, str):
            _write_option(exporter, context, opt, note)