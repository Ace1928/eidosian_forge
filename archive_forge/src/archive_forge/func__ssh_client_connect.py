import os
import re
import time
import atexit
import random
import socket
import hashlib
import binascii
import datetime
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Type, Tuple, Union, Callable, Optional
import libcloud.compute.ssh
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import b
from libcloud.common.base import BaseDriver, Connection, ConnectionKey
from libcloud.compute.ssh import SSHClient, BaseSSHClient, SSHCommandTimeoutError, have_paramiko
from libcloud.common.types import LibcloudError
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet, is_valid_ip_address
def _ssh_client_connect(self, ssh_client, wait_period=1.5, timeout=300):
    """
        Try to connect to the remote SSH server. If a connection times out or
        is refused it is retried up to timeout number of seconds.

        :param ssh_client: A configured SSHClient instance
        :type ssh_client: ``SSHClient``

        :param wait_period: How many seconds to wait between each loop
                            iteration. (default is 1.5)
        :type wait_period: ``int``

        :param timeout: How many seconds to wait before giving up.
                        (default is 300)
        :type timeout: ``int``

        :return: ``SSHClient`` on success
        """
    start = time.time()
    end = start + timeout
    while time.time() < end:
        try:
            ssh_client.connect()
        except SSH_TIMEOUT_EXCEPTION_CLASSES as e:
            message = str(e).lower()
            for fatal_msg in SSH_FATAL_ERROR_MSGS:
                if fatal_msg in message:
                    raise e
            try:
                ssh_client.close()
            except Exception:
                pass
            time.sleep(wait_period)
            continue
        else:
            return ssh_client
    raise LibcloudError(value='Could not connect to the remote SSH ' + 'server. Giving up.', driver=self)