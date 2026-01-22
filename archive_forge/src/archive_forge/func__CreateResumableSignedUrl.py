from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import json
import time
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.diagnose import diagnose_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def _CreateResumableSignedUrl(self, service_account, expiration, bucket, filepath):
    """Make a resumable signed url using the SignBlob API of a Service Account.

    This creates a signed url that can be used by another program to upload a
    single file to the specified bucket with the specified file name.

    Args:
      service_account: The email of a service account that has permissions to
        sign a blob and create files within GCS Buckets.
      expiration: The time at which the returned signed url becomes invalid,
        measured in seconds since the epoch.
      bucket: The name of the bucket the signed url will point to.
      filepath: The name or relative path the file will have within the bucket
        once uploaded.

    Returns:
      A string url that can be used until its expiration to upload a file.
    """
    url_data = six.ensure_binary('POST\n\n\n{0}\nx-goog-resumable:start\n/{1}/{2}'.format(expiration, bucket, filepath))
    signed_blob = ''
    try:
        signed_blob = self._diagnose_client.SignBlob(service_account, url_data)
    except HttpError as e:
        if e.status_code == 403:
            log.Print(_SERVICE_ACCOUNT_TOKEN_CREATOR_ROLE_MISSING_MSG)
        raise
    signature = six.ensure_binary(signed_blob)
    encoded_sig = base64.b64encode(signature)
    url = 'https://storage.googleapis.com/{0}/{1}?GoogleAccessId={2}&Expires={3}&Signature={4}'
    url_suffix = six.moves.urllib.parse.quote_plus(encoded_sig)
    return url.format(bucket, filepath, service_account, expiration, url_suffix)