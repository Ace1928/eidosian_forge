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
def UploadFiles(files_to_upload, num_threads=DEFAULT_NUM_THREADS, show_progress_bar=False):
    """Upload the given files to the given Cloud Storage URLs.

  Uses the appropriate parallelism (multi-process, multi-thread, both, or
  synchronous).

  Args:
    files_to_upload: list of FileUploadTask
    num_threads: int (optional), the number of threads to use.
    show_progress_bar: bool. If true, show a progress bar to the users when
      uploading files.
  """
    num_files = len(files_to_upload)
    if show_progress_bar:
        label = 'Uploading {} {} to Google Cloud Storage'.format(num_files, text.Pluralize(num_files, 'file'))
    else:
        label = None
    ExecuteTasks(files_to_upload, num_threads, label)