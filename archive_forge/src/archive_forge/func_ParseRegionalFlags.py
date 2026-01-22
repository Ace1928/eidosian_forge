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
def ParseRegionalFlags(args, resources, message=None):
    """Make Frontend suitable for RegionalLister argument namespace.

  Generated client-side filter is stored to args.filter.

  Args:
    args: The argument namespace of RegionalLister.
    resources: resources.Registry, The resource registry
    message: The response resource proto message for the request.

  Returns:
    Frontend initialized with information from RegionalLister argument
    namespace.
  """
    frontend = _GetBaseListerFrontendPrototype(args, message=message)
    filter_expr = frontend.filter
    if args.regions:
        filter_expr, scope_set = _TranslateRegionsFlag(args, resources)
    elif args.filter and 'region' in args.filter:
        scope_set = _TranslateRegionsFilters(args, resources)
    else:
        scope_set = AllScopes([resources.Parse(properties.VALUES.core.project.GetOrFail(), collection='compute.projects')], zonal=False, regional=True)
    return _Frontend(filter_expr, frontend.max_results, scope_set)