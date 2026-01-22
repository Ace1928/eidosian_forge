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
def ListDir(self, path):
    if self._IsRoot(path):
        for b in self._client.ListBuckets(project=properties.VALUES.core.project.Get(required=True)):
            yield b.name
        return
    obj_ref = storage_util.ObjectReference.FromUrl(path, allow_empty_object=True)
    self._LoadObjectsIfMissing(obj_ref.bucket_ref)
    dir_name = self._GetDirString(obj_ref.name)
    parent_dir_length = len(dir_name)
    seen = set()
    for obj_name in self._objects[obj_ref.bucket]:
        if obj_name.startswith(dir_name):
            suffix = obj_name[parent_dir_length:]
            result = suffix.split(self._sep)[0]
            if result not in seen:
                seen.add(result)
                yield result