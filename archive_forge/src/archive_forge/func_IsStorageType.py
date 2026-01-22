from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def IsStorageType(name):
    return name in [t.name for t in _STORAGE_TYPES]