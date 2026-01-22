import gzip
import io
import tarfile
import sys
import os.path
from pathlib import Path
from debian.arfile import ArFile, ArError, ArMember     # pylint: disable=unused-import
from debian.changelog import Changelog
from debian.deb822 import Deb822
def _custom_decompress(command_list):
    try:
        import subprocess
        import signal
        proc = subprocess.Popen(command_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=False, preexec_fn=lambda: signal.signal(signal.SIGPIPE, signal.SIG_DFL))
    except (OSError, ValueError) as e:
        raise DebError("error while running command '%s' as subprocess: '%s'" % (' '.join(command_list), e))
    data = proc.communicate(self.__member.read())[0]
    if proc.returncode != 0:
        raise DebError("command '%s' has failed with code '%s'" % (' '.join(command_list), proc.returncode))
    return io.BytesIO(data)