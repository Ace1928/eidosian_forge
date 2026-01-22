from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc.shared_messages import (
from googlecloudsdk.command_lib.dataproc.shared_messages import (
from googlecloudsdk.command_lib.util.args import labels_util
Creates a Batch message from given args.

    Create a Batch message from given arguments. Only the arguments added in
    AddAddArguments are handled. User need to provide bath job type specific
    message during message creation.

    Args:
      args: Parsed argument.
      batch_job: Batch type job instance.

    Returns:
      A Batch message instance.

    Raises:
      AttributeError: When batch_job is invalid.
    