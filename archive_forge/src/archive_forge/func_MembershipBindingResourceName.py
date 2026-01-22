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
def MembershipBindingResourceName(args):
    """Gets a Membership-Binding resource name from a resource argument.

  Assumes the argument is called BINDING.

  Args:
    args: arguments provided to a command, including a Binding resource arg

  Returns:
    The Binding resource name (e.g.
    projects/x/locations/l/memberships/y/bindings/z)
  """
    return args.CONCEPTS.binding.Parse().RelativeName()