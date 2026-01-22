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
def _RecursiveDirList(self, dir_path):
    for n in self.ListDir(dir_path):
        path = self.Join(dir_path, n)
        yield path
        for x in self._RecursiveDirList(path):
            yield x