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
def FormatListRequests(service, project, scopes, scope_name, filter_expr):
    """Helper for generating list requests."""
    requests = []
    if scopes:
        for scope in scopes:
            request = service.GetRequestType('List')(filter=filter_expr, project=project, maxResults=constants.MAX_RESULTS_PER_PAGE)
            setattr(request, scope_name, scope)
            requests.append((service, 'List', request))
    elif not scope_name:
        requests.append((service, 'List', service.GetRequestType('List')(filter=filter_expr, project=project, maxResults=constants.MAX_RESULTS_PER_PAGE)))
    else:
        request_message = service.GetRequestType('AggregatedList')
        input_params = {}
        if hasattr(request_message, 'includeAllScopes'):
            input_params['includeAllScopes'] = True
        if hasattr(request_message, 'returnPartialSuccess'):
            input_params['returnPartialSuccess'] = True
        if _AllowPartialError():
            requests.append((service, 'AggregatedList', request_message(filter=filter_expr, project=project, maxResults=constants.MAX_RESULTS_PER_PAGE, **input_params)))
        else:
            requests.append((service, 'AggregatedList', request_message(filter=filter_expr, project=project, maxResults=constants.MAX_RESULTS_PER_PAGE)))
    return requests