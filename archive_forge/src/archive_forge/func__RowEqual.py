from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import fnmatch
import json
import os
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import metadata_table
from googlecloudsdk.core.cache import persistent_cache_base
from googlecloudsdk.core.util import files
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _RowEqual(self, a, b):
    """Returns True if rows a and b have the same key."""
    return a[:self.keys] == b[:self.keys]