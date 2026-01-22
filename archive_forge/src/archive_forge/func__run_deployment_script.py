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
def _run_deployment_script(self, task, node, ssh_client, max_tries=3):
    """
        Run the deployment script on the provided node. At this point it is
        assumed that SSH connection has already been established.

        :param task: Deployment task to run.
        :type task: :class:`Deployment`

        :param node: Node to run the task on.
        :type node: ``Node``

        :param ssh_client: A configured and connected SSHClient instance.
        :type ssh_client: :class:`SSHClient`

        :param max_tries: How many times to retry if a deployment fails
                          before giving up. (default is 3)
        :type max_tries: ``int``

        :rtype: :class:`.Node`
        :return: ``Node`` Node instance on success.
        """
    tries = 0
    while tries < max_tries:
        try:
            node = task.run(node, ssh_client)
        except SSHCommandTimeoutError as e:
            raise e
        except Exception as e:
            tries += 1
            if 'ssh session not active' in str(e).lower():
                try:
                    ssh_client.close()
                except Exception:
                    pass
                timeout = int(ssh_client.timeout) if ssh_client.timeout else 10
                ssh_client = self._ssh_client_connect(ssh_client=ssh_client, timeout=timeout)
            if tries >= max_tries:
                tb = traceback.format_exc()
                raise LibcloudError(value='Failed after %d tries: %s.\n%s' % (max_tries, str(e), tb), driver=self)
        else:
            ssh_client.close()
            return node
    return node