from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import mimetypes
import os
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from apitools.base.py import transfer
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions as core_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import scaled_integer
import six
def ReadObject(self, object_ref):
    """Read a file from the given Cloud Storage bucket.

    Args:
      object_ref: storage_util.ObjectReference, The object to read from.

    Raises:
      BadFileException if the file read is not successful.

    Returns:
      file-like object containing the data read.
    """
    data = io.BytesIO()
    chunksize = self._GetChunkSize()
    download = transfer.Download.FromStream(data, chunksize=chunksize)
    download.bytes_http = transports.GetApitoolsTransport(response_encoding=None)
    get_req = self.messages.StorageObjectsGetRequest(bucket=object_ref.bucket, object=object_ref.object)
    log.info('Reading [%s]', object_ref)
    try:
        self.client.objects.Get(get_req, download=download)
    except api_exceptions.HttpError as err:
        raise exceptions.BadFileException('Could not read [{object_}]. Please retry: {err}'.format(object_=object_ref, err=http_exc.HttpException(err)))
    data.seek(0)
    return data