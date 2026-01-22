import optparse
import os
import sys
from subprocess import PIPE, CalledProcessError, Popen
from breezy import osutils
from breezy.tests import ssl_certs
def _openssl(args, input=None):
    """Execute a command in a subproces feeding stdin with the provided input.

    :return: (returncode, stdout, stderr)
    """
    cmd = ['openssl'] + args
    proc = Popen(cmd, stdin=PIPE)
    stdout, stderr = proc.communicate(input.encode('utf-8'))
    if proc.returncode:
        raise CalledProcessError(proc.returncode, cmd)
    return (proc.returncode, stdout, stderr)