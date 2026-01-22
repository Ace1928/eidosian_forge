from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.apigee import base
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import request
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core import log
@classmethod
def GetUploadUrl(cls, identifiers):
    """Apigee API for generating a signed URL for uploading archives.

    This API uses the custom method:
    organizations/*/environments/*/archiveDeployments:generateUploadUrl

    Args:
      identifiers: Dict of identifiers for the request entity path, which must
        include "organizationsId" and "environmentsId".

    Returns:
      A dict of the API response in the form of:
        {"uploadUri": "https://storage.googleapis.com/ ... (full URI)"}

    Raises:
      command_lib.apigee.errors.RequestError if there is an error with the API
        request.
    """
    try:
        return request.ResponseToApiRequest(identifiers, entity_path=cls._entity_path[:-1], entity_collection=cls._entity_path[-1], method=':generateUploadUrl')
    except errors.RequestError as error:
        raise error.RewrittenError('archive deployment', 'get upload url for')