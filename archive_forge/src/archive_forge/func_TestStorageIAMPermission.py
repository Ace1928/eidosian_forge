from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import iam as kms_iam
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.iam import util as iam_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import resources
def TestStorageIAMPermission(bucket, project):
    """Tests storage IAM permission for a given bucket for the user project."""
    client = GetStorageClient()
    messages = GetStorageMessages()
    test_req = messages.StorageBucketsTestIamPermissionsRequest(bucket=bucket, permissions=_GCR_PERMISSION, userProject=project)
    return client.buckets.TestIamPermissions(test_req)