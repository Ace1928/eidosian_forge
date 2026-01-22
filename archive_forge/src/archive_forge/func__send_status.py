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
def _send_status(self, request_number, code, desc=None):
    if desc is None:
        try:
            desc = SFTP_DESC[code]
        except IndexError:
            desc = 'Unknown'
    self._response(request_number, CMD_STATUS, code, desc, '')