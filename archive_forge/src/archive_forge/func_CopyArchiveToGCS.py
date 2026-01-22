from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import os.path
import tarfile
import zipfile
from googlecloudsdk.api_lib.cloudbuild import metric_names
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import files
def CopyArchiveToGCS(self, storage_client, gcs_object, ignore_file=None, hide_logs=False):
    """Copy an archive of the snapshot to GCS.

    Args:
      storage_client: storage_api.StorageClient, The storage client to use for
        uploading.
      gcs_object: storage.objects Resource, The GCS object to write.
      ignore_file: Override .gcloudignore file to specify skip files.
      hide_logs: boolean, not print the status message if the flag is true.

    Returns:
      storage_v1_messages.Object, The written GCS object.
    """
    with metrics.RecordDuration(metric_names.UPLOAD_SOURCE):
        with files.ChDir(self.src_dir):
            with files.TemporaryDirectory() as tmp:
                if gcs_object.Name().endswith('.zip'):
                    archive_path = os.path.join(tmp, 'file.zip')
                    self._MakeZipFile(archive_path)
                else:
                    archive_path = os.path.join(tmp, 'file.tgz')
                    tf = self._MakeTarball(archive_path)
                    tf.close()
                ignore_file_path = os.path.join(self.src_dir, ignore_file or gcloudignore.IGNORE_FILE_NAME)
                if self.any_files_ignored:
                    if os.path.exists(ignore_file_path):
                        log.info('Using ignore file [{}]'.format(ignore_file_path))
                    elif not hide_logs:
                        log.status.Print(_IGNORED_FILE_MESSAGE.format(log_file=log.GetLogFilePath()))
                if not hide_logs:
                    file_type = 'zipfile' if gcs_object.Name().endswith('.zip') else 'tarball'
                    log.status.write('Uploading {file_type} of [{src_dir}] to [gs://{bucket}/{object}]\n'.format(file_type=file_type, src_dir=self.src_dir, bucket=gcs_object.bucket, object=gcs_object.object))
                return storage_client.CopyFileToGCS(archive_path, gcs_object)