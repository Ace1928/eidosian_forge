from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataproc.sessions import session_message_factory
Creates a SessionsCreateRequest message.

    Creates a SessionsCreateRequest message. The factory only handles the
    arguments added in AddArguments function. User needs to provide session
    specific message instance.

    Args:
      args: Parsed arguments.

    Returns:
      SessionsCreateRequest: A configured SessionsCreateRequest.
    