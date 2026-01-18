from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc import local_file_uploader
upload user local files and creates a PySparkBatch message.

    Upload user local files and point URIs to the local files to the uploaded
    URIs.
    Creates a PySparkBatch message from parsed arguments.

    Args:
      args: Parsed arguments.

    Returns:
      PySparkBatch: A PySparkBatch message.

    Raises:
      AttributeError: Bucket is required to upload local files, but not
      specified.
    