from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import gc
import os
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import metadata_table
from googlecloudsdk.core.cache import persistent_cache_base
from googlecloudsdk.core.util import files
import six
from six.moves import range  # pylint: disable=redefined-builtin
import sqlite3
def _DeleteCacheFile(self):
    """Permanently deletes the persistent cache file."""
    try:
        os.remove(self.name)
    except OSError as e:
        if e.errno not in (errno.ENOENT, errno.EISDIR):
            raise