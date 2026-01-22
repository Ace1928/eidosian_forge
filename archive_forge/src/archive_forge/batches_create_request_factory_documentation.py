from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc.batches import batch_message_factory
Creates a BatchesCreateRequest message.

    Creates a BatchesCreateRequest message. The factory only handles the
    arguments added in AddArguments function. User needs to provide job
    specific message instance.

    Args:
      args: Parsed arguments.
      batch_job: A batch job typed message instance.

    Returns:
      BatchesCreateRequest: A configured BatchesCreateRequest.
    