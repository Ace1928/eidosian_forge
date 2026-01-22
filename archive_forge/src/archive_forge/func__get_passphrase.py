import binascii
import os
from oslo_concurrency import processutils
from oslo_log import log as logging
from os_brick.encryptors import base
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def _get_passphrase(self, key):
    """Convert raw key to string."""
    return binascii.hexlify(key).decode('utf-8')