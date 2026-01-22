from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def MakeAnalyzeIamPolicyHttpRequests(args, service, messages, api_version=DEFAULT_API_VERSION):
    """Manually make the analyze IAM policy request."""
    parent = asset_utils.GetParentNameForAnalyzeIamPolicy(args.organization, args.project, args.folder)
    full_resource_name = args.full_resource_name if args.IsSpecified('full_resource_name') else None
    identity = args.identity if args.IsSpecified('identity') else None
    roles = args.roles if args.IsSpecified('roles') else []
    permissions = args.permissions if args.IsSpecified('permissions') else []
    expand_groups = args.expand_groups if args.expand_groups else None
    expand_resources = args.expand_resources if args.expand_resources else None
    expand_roles = args.expand_roles if args.expand_roles else None
    saved_analysis_query = args.saved_analysis_query if args.saved_analysis_query else None
    analyze_service_account_impersonation = args.analyze_service_account_impersonation if args.analyze_service_account_impersonation else None
    include_deny_policy_analysis = args.include_deny_policy_analysis if args.IsKnownAndSpecified('include_deny_policy_analysis') else None
    output_resource_edges = None
    if args.output_resource_edges:
        if not args.show_response:
            raise gcloud_exceptions.InvalidArgumentException('--output-resource-edges', 'Must be set together with --show-response to take effect.')
        output_resource_edges = args.output_resource_edges
    output_group_edges = None
    if args.output_group_edges:
        if not args.show_response:
            raise gcloud_exceptions.InvalidArgumentException('--output-group-edges', 'Must be set together with --show-response to take effect.')
        output_group_edges = args.output_group_edges
    execution_timeout = None
    if args.IsSpecified('execution_timeout'):
        execution_timeout = str(args.execution_timeout) + 's'
    access_time = None
    if args.IsKnownAndSpecified('access_time'):
        access_time = times.FormatDateTime(args.access_time)
    response = service.AnalyzeIamPolicy(messages.CloudassetAnalyzeIamPolicyRequest(analysisQuery_accessSelector_permissions=permissions, analysisQuery_accessSelector_roles=roles, analysisQuery_identitySelector_identity=identity, analysisQuery_options_analyzeServiceAccountImpersonation=analyze_service_account_impersonation, analysisQuery_options_expandGroups=expand_groups, analysisQuery_options_expandResources=expand_resources, analysisQuery_options_expandRoles=expand_roles, analysisQuery_options_includeDenyPolicyAnalysis=include_deny_policy_analysis, analysisQuery_options_outputGroupEdges=output_group_edges, analysisQuery_options_outputResourceEdges=output_resource_edges, analysisQuery_resourceSelector_fullResourceName=full_resource_name, analysisQuery_conditionContext_accessTime=access_time, executionTimeout=execution_timeout, scope=parent, savedAnalysisQuery=saved_analysis_query))
    if not args.show_response:
        return _RenderResponseforAnalyzeIamPolicy(response, analyze_service_account_impersonation, api_version)
    return response