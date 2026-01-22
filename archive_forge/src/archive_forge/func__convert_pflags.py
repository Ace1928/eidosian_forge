import os
import errno
import sys
from hashlib import md5, sha1
from paramiko import util
from paramiko.sftp import (
from paramiko.sftp_si import SFTPServerInterface
from paramiko.sftp_attr import SFTPAttributes
from paramiko.common import DEBUG
from paramiko.server import SubsystemHandler
from paramiko.util import b
from paramiko.sftp import (
from paramiko.sftp_handle import SFTPHandle
def _convert_pflags(self, pflags):
    """convert SFTP-style open() flags to Python's os.open() flags"""
    if pflags & SFTP_FLAG_READ and pflags & SFTP_FLAG_WRITE:
        flags = os.O_RDWR
    elif pflags & SFTP_FLAG_WRITE:
        flags = os.O_WRONLY
    else:
        flags = os.O_RDONLY
    if pflags & SFTP_FLAG_APPEND:
        flags |= os.O_APPEND
    if pflags & SFTP_FLAG_CREATE:
        flags |= os.O_CREAT
    if pflags & SFTP_FLAG_TRUNC:
        flags |= os.O_TRUNC
    if pflags & SFTP_FLAG_EXCL:
        flags |= os.O_EXCL
    return flags