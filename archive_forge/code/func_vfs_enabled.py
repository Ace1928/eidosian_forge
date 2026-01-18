import os
from ... import urlutils
from . import request
def vfs_enabled():
    """Is the VFS enabled ?

    the VFS is disabled when the BRZ_NO_SMART_VFS environment variable is set.

    :return: ``True`` if it is enabled.
    """
    return 'BRZ_NO_SMART_VFS' not in os.environ