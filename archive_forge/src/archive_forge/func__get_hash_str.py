import os
import re
import tempfile
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils.secretutils import md5
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
def _get_hash_str(self, base_str):
    """Return a string that represents hash of base_str (hex format)."""
    if isinstance(base_str, str):
        base_str = base_str.encode('utf-8')
    return md5(base_str, usedforsecurity=False).hexdigest()