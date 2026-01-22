from binascii import hexlify
import errno
import os
import stat
import threading
import time
import weakref
from paramiko import util
from paramiko.channel import Channel
from paramiko.message import Message
from paramiko.common import INFO, DEBUG, o777
from paramiko.sftp import (
from paramiko.sftp_attr import SFTPAttributes
from paramiko.ssh_exception import SSHException
from paramiko.sftp_file import SFTPFile
from paramiko.util import ClosingContextManager, b, u
def _convert_status(self, msg):
    """
        Raises EOFError or IOError on error status; otherwise does nothing.
        """
    code = msg.get_int()
    text = msg.get_text()
    if code == SFTP_OK:
        return
    elif code == SFTP_EOF:
        raise EOFError(text)
    elif code == SFTP_NO_SUCH_FILE:
        raise IOError(errno.ENOENT, text)
    elif code == SFTP_PERMISSION_DENIED:
        raise IOError(errno.EACCES, text)
    else:
        raise IOError(text)