from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import filter_scope_rewriter
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.resource import resource_projector
import six
def _GetBaseListerFrontendPrototype(args, message=None):
    """Make Frontend suitable for BaseLister argument namespace.

  Generated client-side filter is stored to args.filter. Generated server-side
  filter is None. Client-side filter should be processed using
  flags.RewriteFilter before use to take advantage of possible server-side
  filtering.

  Args:
    args: The argument namespace of BaseLister.
    message: The resource proto message.

  Returns:
    Frontend initialized with information from BaseLister argument namespace.
    Server-side filter is None.
  """
    frontend = _GetListCommandFrontendPrototype(args, message=message)
    filter_args = []
    default = args.filter
    if args.filter:
        filter_args.append('(' + args.filter + ')')
    if getattr(args, 'regexp', None):
        filter_args.append('(name ~ "^{}$")'.format(resource_expr_rewrite.BackendBase().Quote(args.regexp)))
    if getattr(args, 'names', None):
        name_regexp = ' '.join([resource_expr_rewrite.BackendBase().Quote(name) for name in args.names if not name.startswith('https://')])
        selflink_regexp = ' '.join([resource_expr_rewrite.BackendBase().Quote(name) for name in args.names if name.startswith('https://')])
        if not selflink_regexp:
            filter_args.append('(name =({}))'.format(name_regexp))
        elif not name_regexp:
            filter_args.append('(selfLink =({}))'.format(selflink_regexp))
        else:
            filter_args.append('((name =({})) OR (selfLink =({})))'.format(name_regexp, selflink_regexp))
    args.filter = ' AND '.join(filter_args) or default
    return _Frontend(None, frontend.max_results, frontend.scope_set)