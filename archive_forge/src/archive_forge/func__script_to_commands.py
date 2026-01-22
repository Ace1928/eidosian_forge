import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
def _script_to_commands(text, file_name=None):
    """Turn a script into a list of commands with their associated IOs.

    Each command appears on a line by itself starting with '$ '. It can be
    associated with an input that will feed it and an expected output.

    Comments starts with '#' until the end of line.
    Empty lines are ignored.

    Input and output are full lines terminated by a '
'.

    Input lines start with '<'.
    Output lines start with nothing.
    Error lines start with '2>'.

    :return: A sequence of ([args], input, output, errors), where the args are
        split in to words, and the input, output, and errors are just strings,
        typically containing newlines.
    """
    commands = []

    def add_command(cmd, input, output, error):
        if cmd is not None:
            if input is not None:
                input = ''.join(input)
            if output is not None:
                output = ''.join(output)
            if error is not None:
                error = ''.join(error)
            commands.append((cmd, input, output, error))
    cmd_cur = None
    cmd_line = 1
    lineno = 0
    input, output, error = (None, None, None)
    text = textwrap.dedent(text)
    lines = text.split('\n')
    if lines and lines[0] == '':
        del lines[0]
    if lines and lines[-1] == '':
        del lines[-1]
    for line in lines:
        lineno += 1
        orig = line
        comment = line.find('#')
        if comment >= 0:
            line = line[0:comment]
            line = line.rstrip()
            if line == '':
                continue
        if line.startswith('$'):
            add_command(cmd_cur, input, output, error)
            cmd_cur = list(split(line[1:]))
            cmd_line = lineno
            input, output, error = (None, None, None)
        elif line.startswith('<'):
            if input is None:
                if cmd_cur is None:
                    raise SyntaxError('No command for that input', (file_name, lineno, 1, orig))
                input = []
            input.append(line[1:] + '\n')
        elif line.startswith('2>'):
            if error is None:
                if cmd_cur is None:
                    raise SyntaxError('No command for that error', (file_name, lineno, 1, orig))
                error = []
            error.append(line[2:] + '\n')
        else:
            if output is None:
                if cmd_cur is None:
                    raise SyntaxError('No command for line {!r}'.format(line), (file_name, lineno, 1, orig))
                output = []
            output.append(line + '\n')
    add_command(cmd_cur, input, output, error)
    return commands