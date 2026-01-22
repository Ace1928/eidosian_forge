import inspect
import os
import sys
import breezy
from . import commands as _mod_commands
from . import errors, help_topics, option
from . import plugin as _mod_plugin
from .i18n import gettext
from .trace import mutter, note
def _command_helps(exporter, plugin_name=None):
    """Extract docstrings from path.

    This respects the Bazaar cmdtable/table convention and will
    only extract docstrings from functions mentioned in these tables.
    """
    for cmd_name in _mod_commands.builtin_command_names():
        command = _mod_commands.get_cmd_object(cmd_name, False)
        if command.hidden:
            continue
        if plugin_name is not None:
            continue
        note(gettext('Exporting messages from builtin command: %s'), cmd_name)
        _write_command_help(exporter, command)
    plugins = _mod_plugin.plugins()
    if plugin_name is not None and plugin_name not in plugins:
        raise errors.BzrError(gettext('Plugin %s is not loaded' % plugin_name))
    core_plugins = {name for name in plugins if plugins[name].path().startswith(breezy.__path__[0])}
    for cmd_name in _mod_commands.plugin_command_names():
        command = _mod_commands.get_cmd_object(cmd_name, False)
        if command.hidden:
            continue
        if plugin_name is not None and command.plugin_name() != plugin_name:
            continue
        if plugin_name is None and command.plugin_name() not in core_plugins:
            continue
        note(gettext('Exporting messages from plugin command: {0} in {1}').format(cmd_name, command.plugin_name()))
        _write_command_help(exporter, command)