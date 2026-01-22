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
def ExpandPaths(self, paths):
    files = set()
    dirs = set()
    for p in paths:
        current_files, current_dirs = self.ExpandPath(p)
        if not current_files and (not current_dirs):
            log.warning('[{}] does not match any paths.'.format(p))
            continue
        files.update(current_files)
        dirs.update(current_dirs)
    return (files, dirs)