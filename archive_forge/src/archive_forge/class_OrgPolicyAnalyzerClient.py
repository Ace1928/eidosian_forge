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
class OrgPolicyAnalyzerClient(object):
    """Client for org policy analysis."""
    _DEFAULT_PAGE_SIZE = 100

    def __init__(self, api_version=DEFAULT_API_VERSION):
        self.message_module = GetMessages(api_version)
        self.service = GetClient(api_version).v1

    def AnalyzeOrgPolicyGovernedResources(self, args):
        """Calls AnalyzeOrgPolicyGovernedResources method."""
        request = self.message_module.CloudassetAnalyzeOrgPolicyGovernedResourcesRequest(scope=args.scope, constraint=args.constraint)
        return list_pager.YieldFromList(self.service, request, method='AnalyzeOrgPolicyGovernedResources', field='governedResources', batch_size=args.page_size or self._DEFAULT_PAGE_SIZE, batch_size_attribute='pageSize', current_token_attribute='pageToken', next_token_attribute='nextPageToken')

    def AnalyzeOrgPolicyGovernedAssets(self, args):
        """Calls AnalyzeOrgPolicyGovernedAssets method."""
        request = self.message_module.CloudassetAnalyzeOrgPolicyGovernedAssetsRequest(scope=args.scope, constraint=args.constraint)
        return list_pager.YieldFromList(self.service, request, method='AnalyzeOrgPolicyGovernedAssets', field='governedAssets', batch_size=args.page_size or self._DEFAULT_PAGE_SIZE, batch_size_attribute='pageSize', current_token_attribute='pageToken', next_token_attribute='nextPageToken')

    def AnalyzeOrgPolicyGovernedContainers(self, args):
        """Calls AnalyzeOrgPolicyGovernedContainers method."""
        request = self.message_module.CloudassetAnalyzeOrgPolicyGovernedContainersRequest(scope=args.scope, constraint=args.constraint)
        return list_pager.YieldFromList(self.service, request, method='AnalyzeOrgPolicyGovernedContainers', field='governedContainers', batch_size=args.page_size or self._DEFAULT_PAGE_SIZE, batch_size_attribute='pageSize', current_token_attribute='pageToken', next_token_attribute='nextPageToken')

    def AnalyzeOrgPolicies(self, args):
        """Calls AnalyzeOrgPolicies method."""
        request = self.message_module.CloudassetAnalyzeOrgPoliciesRequest(scope=args.scope, constraint=args.constraint)
        return list_pager.YieldFromList(self.service, request, method='AnalyzeOrgPolicies', field='orgPolicyResults', batch_size=args.page_size or self._DEFAULT_PAGE_SIZE, batch_size_attribute='pageSize', current_token_attribute='pageToken', next_token_attribute='nextPageToken')