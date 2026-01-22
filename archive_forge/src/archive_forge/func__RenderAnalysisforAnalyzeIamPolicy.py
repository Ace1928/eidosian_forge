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
def _RenderAnalysisforAnalyzeIamPolicy(analysis, api_version=DEFAULT_API_VERSION):
    """Renders the analysis query and results of the AnalyzeIamPolicy request."""
    for analysis_result in analysis.analysisResults:
        entry = {}
        policy = {'attachedResource': analysis_result.attachedResourceFullName, 'binding': analysis_result.iamBinding}
        entry['policy'] = policy
        entry['ACLs'] = []
        for acl in analysis_result.accessControlLists:
            acls = {}
            acls['identities'] = analysis_result.identityList.identities
            acls['accesses'] = acl.accesses
            acls['resources'] = acl.resources
            if api_version == DEFAULT_API_VERSION and acl.conditionEvaluation:
                acls['conditionEvaluationValue'] = acl.conditionEvaluation.evaluationValue
            entry['ACLs'].append(acls)
        yield entry
    for deny_analysis_result in analysis.denyAnalysisResults:
        entry = {}
        access_tuple = {'resource': deny_analysis_result.accessTuple.resource, 'access': deny_analysis_result.accessTuple.access, 'identity': deny_analysis_result.accessTuple.identity}
        entry['access_tuple'] = access_tuple
        entry['deny_details'] = []
        for detail in deny_analysis_result.denyDetails:
            deny_detail = {}
            deny_detail['deny_rule'] = detail.denyRule
            deny_detail['resources'] = detail.resources
            deny_detail['identities'] = detail.identities
            deny_detail['accesses'] = detail.accesses
            deny_detail['exception_identities'] = detail.exceptionIdentities
            entry['deny_details'].append(deny_detail)
        yield entry