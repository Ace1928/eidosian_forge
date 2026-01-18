import hashlib
import logging
import os
import socket
from oslo_config import cfg
from glance_store._drivers.cinder import base
from glance_store.common import cinder_utils
from glance_store.common import fs_mount as mount
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _
Returns string representing SHA256 hash of base_str in hex format.

        If base_str is a Unicode string, encode it to UTF-8.
        