from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def UploadFileToGcs(self, path, mimetype=None, destination_object=None):
    """Upload a file to the GCS results bucket using the storage API.

    Args:
      path: str, the absolute or relative path of the file to upload. File may
        be in located in GCS or the local filesystem.
      mimetype: str, the MIME type (aka Content-Type) that should be applied to
        files being copied from a non-GCS source to GCS. MIME types for GCS->GCS
        file uploads are not modified.
      destination_object: str, the destination object path in GCS to upload to,
        if it's different than the base name of the path argument.

    Raises:
      BadFileException if the file upload is not successful.
    """
    log.status.Print('Uploading [{f}] to Firebase Test Lab...'.format(f=path))
    try:
        if path.startswith(GCS_PREFIX):
            file_bucket, file_obj = _SplitBucketAndObject(path)
            copy_req = self._storage_messages.StorageObjectsCopyRequest(sourceBucket=file_bucket, sourceObject=file_obj, destinationBucket=self._results_bucket, destinationObject='{obj}/{name}'.format(obj=self._gcs_object_name, name=destination_object or os.path.basename(file_obj)))
            self._storage_client.objects.Copy(copy_req)
        else:
            try:
                file_size = os.path.getsize(path)
            except os.error:
                raise exceptions.BadFileException('[{0}] not found or not accessible'.format(path))
            src_obj = self._storage_messages.Object(size=file_size)
            try:
                upload = transfer.Upload.FromFile(path, mime_type=mimetype)
            except apitools_exceptions.InvalidUserInputError:
                upload = transfer.Upload.FromFile(path, mime_type='application/octet-stream')
            insert_req = self._storage_messages.StorageObjectsInsertRequest(bucket=self._results_bucket, name='{obj}/{name}'.format(obj=self._gcs_object_name, name=destination_object or os.path.basename(path)), object=src_obj)
            response = self._storage_client.objects.Insert(insert_req, upload=upload)
            if response.size != file_size:
                raise exceptions.BadFileException('Cloud storage upload failure: Insert response.size={0} bytes but [{1}] contains {2} bytes.\nInsert response: {3}'.format(response.size, path, file_size, repr(response)))
    except apitools_exceptions.HttpError as err:
        raise exceptions.BadFileException('Could not copy [{f}] to [{gcs}] {e}.'.format(f=path, gcs=self.gcs_results_root, e=util.GetError(err)))