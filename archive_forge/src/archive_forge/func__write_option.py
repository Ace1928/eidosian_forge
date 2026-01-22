import inspect
import os
import sys
import breezy
from . import commands as _mod_commands
from . import errors, help_topics, option
from . import plugin as _mod_plugin
from .i18n import gettext
from .trace import mutter, note
def _write_option(exporter, context, opt, note):
    if getattr(opt, 'hidden', False):
        return
    optname = opt.name
    if getattr(opt, 'title', None):
        exporter.poentry_in_context(context, opt.title, f'title of {optname!r} {note}')
    for name, _, _, helptxt in opt.iter_switches():
        if name != optname:
            if opt.is_hidden(name):
                continue
            name = '='.join([optname, name])
        if helptxt:
            exporter.poentry_in_context(context, helptxt, f'help of {name!r} {note}')