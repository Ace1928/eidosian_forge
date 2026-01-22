from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import enum
import functools
import re
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.api_lib.compute.regions import service as regions_service
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import scope_prompter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.util import text
import six
def RewriteFilter(args, message=None, frontend_fields=None):
    """Rewrites args.filter into client and server filter expression strings.

  Usage:

    args.filter, request_filter = flags.RewriteFilter(args)

  Args:
    args: The parsed args namespace containing the filter expression args.filter
      and display_info.
    message: The response resource message proto for the request.
    frontend_fields: A set of dotted key names supported client side only.

  Returns:
    A (client_filter, server_filter) tuple of filter expression strings.
    None means the filter does not need to applied on the respective
    client/server side.
  """
    if not args.filter:
        return (None, None)
    display_info = args.GetDisplayInfo()
    defaults = resource_projection_spec.ProjectionSpec(symbols=display_info.transforms, aliases=display_info.aliases)
    client_filter, server_filter = filter_rewrite.Rewriter(message=message, frontend_fields=frontend_fields).Rewrite(args.filter, defaults=defaults)
    log.info('client_filter=%r server_filter=%r', client_filter, server_filter)
    return (client_filter, server_filter)