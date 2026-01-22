from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.core.console import console_io
import six
def HasLocalFiles(files):
    """Determines whether files argument has local files.

  Args:
    files: A dictionary of lists of files to check.

  Returns:
    True if at least one of the files is local.

  Example:
    GetLocalFiles({'jar':['my-jar.jar', gs://my-bucket/my-gcs-jar.jar]}) -> True
  """
    for _, uris in files.items():
        for uri in uris:
            if _IsLocal(uri):
                return True
    return False