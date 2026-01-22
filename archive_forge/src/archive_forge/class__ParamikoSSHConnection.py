import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
class _ParamikoSSHConnection(SSHConnection):
    """An SSH connection via paramiko."""

    def __init__(self, channel):
        self.channel = channel

    def get_sock_or_pipes(self):
        return ('socket', self.channel)

    def close(self):
        return self.channel.close()