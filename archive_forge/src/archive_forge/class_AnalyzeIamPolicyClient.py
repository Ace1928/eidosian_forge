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
class AnalyzeIamPolicyClient(object):
    """Client for IAM policy analysis."""

    def __init__(self, api_version=DEFAULT_API_VERSION):
        self.api_version = api_version
        self.client = GetClient(api_version)
        self.service = self.client.v1

    def Analyze(self, args):
        """Calls MakeAnalyzeIamPolicy method."""
        messages = self.EncodeMessages(args)
        return MakeAnalyzeIamPolicyHttpRequests(args, self.service, messages, self.api_version)

    def EncodeMessages(self, args):
        """Adds custom encoding for MakeAnalyzeIamPolicy request."""
        messages = GetMessages(self.api_version)

        def AddCustomJsonFieldMapping(prefix, suffix):
            field = _IAM_POLICY_ANALYZER_VERSION_DICT_JSON[self.api_version][prefix] + suffix
            encoding.AddCustomJsonFieldMapping(messages.CloudassetAnalyzeIamPolicyRequest, field, field.replace('_', '.'))
        AddCustomJsonFieldMapping('resource_selector', '_fullResourceName')
        AddCustomJsonFieldMapping('identity_selector', '_identity')
        AddCustomJsonFieldMapping('access_selector', '_roles')
        AddCustomJsonFieldMapping('access_selector', '_permissions')
        AddCustomJsonFieldMapping('options', '_expandGroups')
        AddCustomJsonFieldMapping('options', '_expandResources')
        AddCustomJsonFieldMapping('options', '_expandRoles')
        AddCustomJsonFieldMapping('options', '_outputResourceEdges')
        AddCustomJsonFieldMapping('options', '_outputGroupEdges')
        AddCustomJsonFieldMapping('options', '_analyzeServiceAccountImpersonation')
        if args.IsKnownAndSpecified('include_deny_policy_analysis'):
            AddCustomJsonFieldMapping('options', '_includeDenyPolicyAnalysis')
        if args.IsKnownAndSpecified('access_time'):
            AddCustomJsonFieldMapping('condition_context', '_accessTime')
        return messages