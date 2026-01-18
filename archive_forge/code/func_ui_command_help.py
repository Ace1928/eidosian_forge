import inspect
import re
import six
def ui_command_help(self, topic=None):
    """
        Displays the manual page for a topic, or list available topics.
        """
    commands = self.list_commands()
    if topic is None:
        msg = self.shell.con.dedent(self.help_intro)
        msg += self.shell.con.dedent('\n\n                                   AVAILABLE COMMANDS\n                                   ==================\n                                   The following commands are available in the\n                                   current path:\n\n                                   ')
        for command in commands:
            msg += '  - %s\n' % self.get_command_syntax(command)[0]
        self.shell.con.epy_write(msg)
        return
    if topic not in commands:
        raise ExecutionError('Cannot find help topic %s.' % topic)
    syntax, comments, defaults = self.get_command_syntax(topic)
    msg = self.shell.con.dedent('\n                             SYNTAX\n                             ======\n                             %s\n\n                             ' % syntax)
    for comment in comments:
        msg += comment + '\n'
    if defaults:
        msg += self.shell.con.dedent('\n                                  DEFAULT VALUES\n                                  ==============\n                                  %s\n\n                                  ' % defaults)
    msg += self.shell.con.dedent('\n                              DESCRIPTION\n                              ===========\n                              ')
    msg += self.get_command_description(topic)
    msg += '\n'
    self.shell.con.epy_write(msg)