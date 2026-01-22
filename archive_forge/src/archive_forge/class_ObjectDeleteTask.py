from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
import six
class ObjectDeleteTask(Task):
    """Self-contained representation of an object to delete.

  Attributes:
    obj_ref: storage_util.ObjectReference, The object to delete.
  """

    def __init__(self, obj_ref):
        self.obj_ref = obj_ref

    def __str__(self):
        return 'Delete: {}'.format(self.obj_ref.ToUrl())

    def __repr__(self):
        return 'ObjectDeleteTask(object={obj}'.format(obj=self.obj_ref.ToUrl())

    def __hash__(self):
        return hash(self.obj_ref)

    def Execute(self, callback=None):
        """Complete one ObjectDeleteTask (safe to run in parallel)."""
        storage_client = storage_api.StorageClient()
        retry.Retryer(max_retrials=3).RetryOnException(storage_client.DeleteObject, args=(self.obj_ref,))
        if callback:
            callback()