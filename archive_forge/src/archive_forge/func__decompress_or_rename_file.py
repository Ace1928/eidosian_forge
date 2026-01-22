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
def _decompress_or_rename_file(source_resource, temporary_file_path, final_file_path, do_not_decompress_flag=False, server_encoding=None):
    """Converts temporary file to final form by decompressing or renaming.

  Args:
    source_resource (ObjectResource): May contain encoding metadata.
    temporary_file_path (str): File path to unzip or rename.
    final_file_path (str): File path to write final file to.
    do_not_decompress_flag (bool): User flag that blocks decompression.
    server_encoding (str|None): Server-reported `content-encoding` of file.

  Returns:
    (bool) True if file was decompressed or renamed, and
      False if file did not exist.
  """
    if not os.path.exists(temporary_file_path):
        return False
    if gzip_util.decompress_gzip_if_necessary(source_resource, temporary_file_path, final_file_path, do_not_decompress_flag, server_encoding):
        os.remove(temporary_file_path)
    else:
        os.rename(temporary_file_path, final_file_path)
    return True