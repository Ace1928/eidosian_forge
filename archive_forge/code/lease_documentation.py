from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import list_formats
from googlecloudsdk.command_lib.tasks import parsers
Leases a list of tasks and displays them.

  Each task returned from this command will have its schedule time changed
  based on the lease duration specified. A task that has been returned by
  calling this command will not be returned in a different call before its
  schedule time. After the work associated with a task is finished, the lease
  holder should call `gcloud tasks acknowledge` on the task.
  