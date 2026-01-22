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
class UuidMixin:
    """
    Mixin class for get_uuid function.
    """

    def __init__(self) -> None:
        self._uuid = None

    def get_uuid(self):
        """
        Unique hash for a node, node image, or node size

        The hash is a function of an SHA1 hash of the node, node image,
        or node size's ID and its driver which means that it should be
        unique between all objects of its type.
        In some subclasses there is no ID available so the public IP
        address is used.  This means that, unlike a properly done system
        UUID, the same UUID may mean a different system install at a
        different time

        >>> from libcloud.compute.drivers.dummy import DummyNodeDriver
        >>> driver = DummyNodeDriver(0)
        >>> node = driver.create_node()
        >>> node.get_uuid()
        'd3748461511d8b9b0e0bfa0d4d3383a619a2bb9f'

        Note, for example, that this example will always produce the
        same UUID!

        :rtype: ``str``
        """
        if not self._uuid:
            self._uuid = hashlib.sha1(b('{}:{}'.format(self.id, self.driver.type))).hexdigest()
        return self._uuid

    @property
    def uuid(self):
        return self.get_uuid()