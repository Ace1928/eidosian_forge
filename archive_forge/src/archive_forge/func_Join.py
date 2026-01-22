from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import fnmatch
import os
import re
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def Join(self, path1, path2):
    if self._IsRoot(path1):
        return 'gs://' + path2.lstrip(self._sep)
    return path1.rstrip(self._sep) + self._sep + path2.lstrip(self._sep)