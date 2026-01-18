import errno
import os
import sys
import tempfile
from subprocess import PIPE, Popen
from .errors import BzrError, NoDiff3
from .textfile import check_text_path
def write_to_cmd(args, input=''):
    """Spawn a process, and wait for the result

    If the process is killed, an exception is raised

    :param args: The command line, the first entry should be the program name
    :param input: [optional] The text to send the process on stdin
    :return: (stdout, stderr, status)
    """
    process = Popen(args, bufsize=len(input), stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=_do_close_fds)
    stdout, stderr = process.communicate(input)
    status = process.wait()
    if status < 0:
        raise Exception('%s killed by signal %i' % (args[0], -status))
    return (stdout, stderr, status)