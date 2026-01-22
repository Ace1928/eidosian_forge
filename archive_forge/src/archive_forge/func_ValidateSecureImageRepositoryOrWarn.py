from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import functools
import json
import re
from apitools.base.py import base_api
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.functions.v1 import exceptions
from googlecloudsdk.api_lib.functions.v1 import operations
from googlecloudsdk.api_lib.functions.v2 import util as v2_util
from googlecloudsdk.api_lib.storage import storage_api as gcs_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as base_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.generated_clients.apis.cloudfunctions.v1 import cloudfunctions_v1_messages
import six.moves.http_client
def ValidateSecureImageRepositoryOrWarn(region_name, project_id):
    """Validates image repository. Yields security and deprecation warnings.

  Args:
    region_name: String name of the region to which the function is deployed.
    project_id: String ID of the Cloud project.
  """
    _AddGcrDeprecationWarning()
    gcr_bucket_url = GetStorageBucketForGcrRepository(region_name, project_id)
    try:
        gcr_host_policy = gcs_api.StorageClient().GetIamPolicy(storage_util.BucketReference.FromUrl(gcr_bucket_url))
        if gcr_host_policy and iam_util.BindingInPolicy(gcr_host_policy, 'allUsers', 'roles/storage.objectViewer'):
            log.warning("The Container Registry repository that stores this function's image is public. This could pose the risk of disclosing sensitive data. To mitigate this, either use Artifact Registry ('--docker-registry=artifact-registry' flag) or change this setting in Google Container Registry.\n")
    except apitools_exceptions.HttpError:
        log.warning("Secuirty check for Container Registry repository that stores this function's image has not succeeded. To mitigate risks of disclosing sensitive data, it is recommended to keep your repositories private. This setting can be verified in Google Container Registry.\n")