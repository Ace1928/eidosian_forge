from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
def UseMaxRequestedPolicyVersion(api_field):
    """Set requestedPolicyVersion to max supported in GetIamPolicy request."""

    def Process(ref, args, request):
        del ref, args
        arg_utils.SetFieldInMessage(request, api_field, iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION)
        return request
    return Process