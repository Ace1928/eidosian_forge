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
class FileRemoteCopyTask(Task):
    """Self-contained representation of a copy between GCS objects.

  Attributes:
    source_obj_ref: storage_util.ObjectReference, The object reference of the
      file to download.
    dest_obj_ref: storage_util.ObjectReference, The object reference to write
      the file to.
  """

    def __init__(self, source_obj_ref, dest_obj_ref):
        self.source_obj_ref = source_obj_ref
        self.dest_obj_ref = dest_obj_ref

    def __str__(self):
        return 'Copy: {} --> {}'.format(self.source_obj_ref.ToUrl(), self.dest_obj_ref.ToUrl())

    def __repr__(self):
        return 'FileRemoteCopyTask(source_path={source_path}, dest_path={dest_path})'.format(source_path=self.source_obj_ref.ToUrl(), dest_path=self.dest_obj_ref.ToUrl())

    def __hash__(self):
        return hash((self.source_obj_ref, self.dest_obj_ref))

    def Execute(self, callback=None):
        storage_client = storage_api.StorageClient()
        retry.Retryer(max_retrials=3).RetryOnException(storage_client.Copy, args=(self.source_obj_ref, self.dest_obj_ref))
        if callback:
            callback()