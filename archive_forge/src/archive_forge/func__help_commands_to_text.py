from . import commands as _mod_commands
from . import errors, help_topics, osutils, plugin, ui, utextwrap
def _help_commands_to_text(topic):
    """Generate the help text for the list of commands"""
    out = []
    if topic == 'hidden-commands':
        hidden = True
    else:
        hidden = False
    names = list(_mod_commands.all_command_names())
    commands = ((n, _mod_commands.get_cmd_object(n)) for n in names)
    shown_commands = [(n, o) for n, o in commands if o.hidden == hidden]
    max_name = max((len(n) for n, o in shown_commands))
    indent = ' ' * (max_name + 1)
    width = osutils.terminal_width()
    if width is None:
        width = osutils.default_terminal_width
    width = width - 1
    for cmd_name, cmd_object in sorted(shown_commands):
        plugin_name = cmd_object.plugin_name()
        if plugin_name is None:
            plugin_name = ''
        else:
            plugin_name = ' [%s]' % plugin_name
        cmd_help = cmd_object.help()
        if cmd_help:
            firstline = cmd_help.split('\n', 1)[0]
        else:
            firstline = ''
        helpstring = '%-*s %s%s' % (max_name, cmd_name, firstline, plugin_name)
        lines = utextwrap.wrap(helpstring, subsequent_indent=indent, width=width, break_long_words=False)
        for line in lines:
            out.append(line + '\n')
    return ''.join(out)