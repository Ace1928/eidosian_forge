from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.apigee import archives as cmd_lib
from googlecloudsdk.command_lib.apigee import defaults
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _GetUploadUrl(self, identifiers):
    """Gets the signed URL for uploading the archive deployment.

    Args:
      identifiers: A dict of resource identifers. Must contain "organizationsId"
        and "environmentsId"

    Returns:
      A str of the upload URL.

    Raises:
      googlecloudsdk.command_lib.apigee.errors.RequestError if the "uploadUri"
        field is not included in the GetUploadUrl response.
    """
    get_upload_url_resp = apigee.ArchivesClient.GetUploadUrl(identifiers)
    if 'uploadUri' not in get_upload_url_resp:
        raise errors.RequestError(resource_type='getUploadUrl', resource_identifier=identifiers, body=get_upload_url_resp, user_help='Please try again.')
    return get_upload_url_resp['uploadUri']