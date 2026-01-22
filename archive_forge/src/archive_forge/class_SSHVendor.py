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
class SSHVendor:
    """Abstract base class for SSH vendor implementations."""

    def connect_sftp(self, username, password, host, port):
        """Make an SSH connection, and return an SFTPClient.

        :param username: an ascii string
        :param password: an ascii string
        :param host: a host name as an ascii string
        :param port: a port number
        :type port: int

        :raises: ConnectionError if it cannot connect.

        :rtype: paramiko.sftp_client.SFTPClient
        """
        raise NotImplementedError(self.connect_sftp)

    def connect_ssh(self, username, password, host, port, command):
        """Make an SSH connection.

        :returns: an SSHConnection.
        """
        raise NotImplementedError(self.connect_ssh)

    def _raise_connection_error(self, host, port=None, orig_error=None, msg='Unable to connect to SSH host'):
        """Raise a SocketConnectionError with properly formatted host.

        This just unifies all the locations that try to raise ConnectionError,
        so that they format things properly.
        """
        raise errors.SocketConnectionError(host=host, port=port, msg=msg, orig_error=orig_error)