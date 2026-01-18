from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc.sessions import (
from googlecloudsdk.command_lib.dataproc.shared_messages import (
from googlecloudsdk.command_lib.dataproc.shared_messages import (
from googlecloudsdk.command_lib.util.args import labels_util
Creates a Session message from given args.

    Create a Session message from given arguments. Only the arguments added in
    AddArguments are handled. Users need to provide session type specific
    message during message creation.

    Args:
      args: Parsed argument.

    Returns:
      A Session message instance.

    Raises:
      AttributeError: When session is invalid.
    