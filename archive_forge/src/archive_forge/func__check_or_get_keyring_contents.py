from __future__ import annotations
import os
import tempfile
import typing
from typing import Any, Optional, Union  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import excutils
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rbd as rbd_privsep
from os_brick import utils
@staticmethod
def _check_or_get_keyring_contents(keyring: Optional[str], cluster_name: str, user: str) -> str:
    try:
        if keyring is None:
            if user:
                keyring_path = '/etc/ceph/%s.client.%s.keyring' % (cluster_name, user)
                with open(keyring_path, 'r') as keyring_file:
                    keyring = keyring_file.read()
            else:
                keyring = ''
        return keyring
    except IOError:
        msg = _('Keyring path %s is not readable.') % keyring_path
        raise exception.BrickException(msg=msg)