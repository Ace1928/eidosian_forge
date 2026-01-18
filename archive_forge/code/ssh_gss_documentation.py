import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__

            You won't get another token if the context is fully established,
            so i set token to None instead of ""
            