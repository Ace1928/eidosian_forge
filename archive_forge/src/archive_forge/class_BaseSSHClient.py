import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
class BaseSSHClient:
    """
    Base class representing a connection over SSH/SCP to a remote node.
    """

    def __init__(self, hostname, port=22, username='root', password=None, key=None, key_files=None, timeout=None):
        """
        :type hostname: ``str``
        :keyword hostname: Hostname or IP address to connect to.

        :type port: ``int``
        :keyword port: TCP port to communicate on, defaults to 22.

        :type username: ``str``
        :keyword username: Username to use, defaults to root.

        :type password: ``str``
        :keyword password: Password to authenticate with or a password used
                           to unlock a private key if a password protected key
                           is used.

        :param key: Deprecated in favor of ``key_files`` argument.

        :type key_files: ``str`` or ``list``
        :keyword key_files: A list of paths to the private key files to use.
        """
        if key is not None:
            message = 'You are using deprecated "key" argument which has been replaced with "key_files" argument'
            warnings.warn(message, DeprecationWarning)
            key_files = key if not key_files else key_files
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.key_files = key_files
        self.timeout = timeout

    def connect(self):
        """
        Connect to the remote node over SSH.

        :return: True if the connection has been successfully established,
                 False otherwise.
        :rtype: ``bool``
        """
        raise NotImplementedError('connect not implemented for this ssh client')

    def put(self, path, contents=None, chmod=None, mode='w'):
        """
        Upload a file to the remote node.

        :type path: ``str``
        :keyword path: File path on the remote node.

        :type contents: ``str``
        :keyword contents: File Contents.

        :type chmod: ``int``
        :keyword chmod: chmod file to this after creation.

        :type mode: ``str``
        :keyword mode: Mode in which the file is opened.

        :return: Full path to the location where a file has been saved.
        :rtype: ``str``
        """
        raise NotImplementedError('put not implemented for this ssh client')

    def putfo(self, path, fo=None, chmod=None):
        """
        Upload file like object to the remote server.

        :param path: Path to upload the file to.
        :type path: ``str``

        :param fo: File like object to read the content from.
        :type fo: File handle or file like object.

        :type chmod: ``int``
        :keyword chmod: chmod file to this after creation.

        :return: Full path to the location where a file has been saved.
        :rtype: ``str``
        """
        raise NotImplementedError('putfo not implemented for this ssh client')

    def delete(self, path):
        """
        Delete/Unlink a file on the remote node.

        :type path: ``str``
        :keyword path: File path on the remote node.

        :return: True if the file has been successfully deleted, False
                 otherwise.
        :rtype: ``bool``
        """
        raise NotImplementedError('delete not implemented for this ssh client')

    def run(self, cmd, timeout=None):
        """
        Run a command on a remote node.

        :type cmd: ``str``
        :keyword cmd: Command to run.

        :return ``list`` of [stdout, stderr, exit_status]
        """
        raise NotImplementedError('run not implemented for this ssh client')

    def close(self):
        """
        Shutdown connection to the remote node.

        :return: True if the connection has been successfully closed, False
                 otherwise.
        :rtype: ``bool``
        """
        raise NotImplementedError('close not implemented for this ssh client')

    def _get_and_setup_logger(self):
        logger = logging.getLogger('libcloud.compute.ssh')
        path = os.getenv('LIBCLOUD_DEBUG')
        if path:
            handler = logging.FileHandler(path)
            handler.setFormatter(ExtraLogFormatter())
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        return logger