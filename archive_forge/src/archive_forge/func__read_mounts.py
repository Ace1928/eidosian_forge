import os
import re
import tempfile
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils.secretutils import md5
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
def _read_mounts(self):
    """Returns a dict of mounts and their mountpoint

        Format reference:
        http://man7.org/linux/man-pages/man5/fstab.5.html
        """
    with open('/proc/mounts', 'r') as mounts:
        lines = [line.split() for line in mounts.read().splitlines() if line.strip()]
        return {line[1]: line[0] for line in lines if line[0] != '#'}