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
def CopyFileFromGCS(self, source_obj_ref, local_path, overwrite=False):
    """Download a file from the given Cloud Storage bucket.

    Args:
      source_obj_ref: storage_util.ObjectReference, the path of the file on GCS
        to download.
      local_path: str, the path of the file to download to. Path must be on the
        local filesystem.
      overwrite: bool, whether or not to overwrite local_path if it already
        exists.

    Raises:
      BadFileException if the file download is not successful.
    """
    chunksize = self._GetChunkSize()
    download = transfer.Download.FromFile(local_path, chunksize=chunksize, overwrite=overwrite)
    download.bytes_http = transports.GetApitoolsTransport(response_encoding=None)
    get_req = self.messages.StorageObjectsGetRequest(bucket=source_obj_ref.bucket, object=source_obj_ref.object)
    gsc_path = '{bucket}/{object_path}'.format(bucket=source_obj_ref.bucket, object_path=source_obj_ref.object)
    log.info('Downloading [{gcs}] to [{local_file}]'.format(local_file=local_path, gcs=gsc_path))
    try:
        self.client.objects.Get(get_req, download=download)
        response = self.client.objects.Get(get_req)
    except api_exceptions.HttpError as err:
        raise exceptions.BadFileException('Could not copy [{gcs}] to [{local_file}]. Please retry: {err}'.format(local_file=local_path, gcs=gsc_path, err=http_exc.HttpException(err)))
    finally:
        download.stream.close()
    file_size = _GetFileSize(local_path)
    if response.size != file_size:
        log.debug('Download size: {0} bytes, but expected size is {1} bytes.'.format(file_size, response.size))
        raise exceptions.BadFileException('Cloud Storage download failure. Downloaded file [{0}] does not match Cloud Storage object. Please retry.'.format(local_path))