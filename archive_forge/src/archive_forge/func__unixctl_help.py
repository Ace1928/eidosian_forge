import sys
import ovs.util
def _unixctl_help(conn, unused_argv, unused_aux):
    reply = 'The available commands are:\n'
    command_names = sorted(commands.keys())
    for name in command_names:
        reply += '  '
        usage = commands[name].usage
        if usage:
            reply += '%-23s %s' % (name, usage)
        else:
            reply += name
        reply += '\n'
    conn.reply(reply)