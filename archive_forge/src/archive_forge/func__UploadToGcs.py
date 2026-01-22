from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os.path
import string
import uuid
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.images import flags
from googlecloudsdk.command_lib.compute.images import os_choices
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
import six
def _UploadToGcs(self, is_async, local_path, daisy_bucket, image_uuid):
    """Uploads a local file to GCS. Returns the gs:// URI to that file."""
    file_name = os.path.basename(local_path).replace(' ', '-')
    dest_path = 'gs://{0}/tmpimage/{1}-{2}'.format(daisy_bucket, image_uuid, file_name)
    if is_async:
        log.status.Print('Async: After upload is complete, your image will be imported from Cloud Storage asynchronously.')
    with progress_tracker.ProgressTracker('Copying [{0}] to [{1}]'.format(local_path, dest_path)):
        return self._UploadToGcsStorageApi(local_path, dest_path)