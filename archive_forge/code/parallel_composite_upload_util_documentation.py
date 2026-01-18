from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import scaled_integer
Checks if parallel composite upload should be performed.

  Logs tailored warning based on user configuration and the context
  of the operation.
  Informs user about configuration options they may want to set.
  In order to avoid repeated warning raised for each task,
  this function updates the storage/parallel_composite_upload_enabled
  so that the warnings are logged only once.

  Args:
    source_resource (FileObjectResource): The source file
      resource to be uploaded.
    destination_resource(CloudResource|UnknownResource):
      Destination resource to which the files should be uploaded.
    user_request_args (UserRequestArgs|None): Values for RequestConfig.

  Returns:
    True if the parallel composite upload can be performed. However, this does
    not guarantee that parallel composite upload will be performed as the
    parallelism check can happen only after the task executor starts running
    because it sets the process_count and thread_count. We also let the task
    determine the component count.
  