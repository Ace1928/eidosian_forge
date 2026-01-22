import time
import breezy
import breezy.commands
import breezy.help
import breezy.help_topics
import breezy.osutils
def _get_commands_section(registry, title='Commands', hdg_level1='#', hdg_level2='=', output_dir=None):
    """Build the commands reference section of the manual."""
    file_per_topic = output_dir is not None
    lines = [title, hdg_level1 * len(title), '']
    if file_per_topic:
        lines.extend(['.. toctree::', '   :maxdepth: 1', ''])
    cmds = sorted(breezy.commands.builtin_command_names())
    for cmd_name in cmds:
        cmd_object = breezy.commands.get_cmd_object(cmd_name)
        if cmd_object.hidden:
            continue
        heading = cmd_name
        underline = hdg_level2 * len(heading)
        text = cmd_object.get_help_text(plain=False, see_also_as_links=True)
        help = '{}\n{}\n\n{}\n\n'.format(heading, underline, text)
        if file_per_topic:
            topic_id = _dump_text(output_dir, cmd_name, help)
            lines.append('   %s' % topic_id)
        else:
            lines.append(help)
    return '\n' + '\n'.join(lines) + '\n'