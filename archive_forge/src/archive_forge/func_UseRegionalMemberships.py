from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import util as cmd_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def UseRegionalMemberships(track=None):
    """Returns whether regional memberships should be included.

  This will be updated as regionalization is released, and eventually deleted
  when it is fully rolled out.

  Args:
    track: The release track of the command

  Returns:
    A bool, whether regional memberships are supported for the release track in
    the active environment
  """
    return track is calliope_base.ReleaseTrack.ALPHA and cmd_util.APIEndpoint() == cmd_util.AUTOPUSH_API