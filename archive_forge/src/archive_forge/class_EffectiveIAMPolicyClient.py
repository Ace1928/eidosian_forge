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
class EffectiveIAMPolicyClient(object):
    """Client for Effective IAM Policy analysis."""

    def __init__(self, api_version=DEFAULT_API_VERSION):
        self.message_module = GetMessages(api_version)
        self.service = GetClient(api_version).effectiveIamPolicies

    def BatchGetEffectiveIAMPolicies(self, args):
        """Calls BatchGetEffectiveIAMPolicies method."""
        request = self.message_module.CloudassetEffectiveIamPoliciesBatchGetRequest(names=args.names, scope=args.scope)
        return self.service.BatchGet(request)