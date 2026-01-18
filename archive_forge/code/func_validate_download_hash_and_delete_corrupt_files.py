from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import symlink_util
from googlecloudsdk.command_lib.storage import tracker_file_util
def validate_download_hash_and_delete_corrupt_files(download_path, source_hash, destination_hash):
    """Confirms hashes match for copied objects.

  Args:
    download_path (str): URL of object being validated.
    source_hash (str): Hash of source object.
    destination_hash (str): Hash of downloaded object.

  Raises:
    HashMismatchError: Hashes are not equal.
  """
    try:
        hash_util.validate_object_hashes_match(download_path, source_hash, destination_hash)
    except errors.HashMismatchError:
        os.remove(download_path)
        tracker_file_util.delete_download_tracker_files(storage_url.storage_url_from_string(download_path))
        raise