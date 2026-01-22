from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import util as functions_api_util
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddArtifactsGCSBucketFlag(parser, hidden=False):
    """Add --artifacts-gcs-bucket flag."""
    parser.add_argument('--artifacts-gcs-bucket', hidden=hidden, help='user-defined location of Cloud Build logs, artifacts, and Terraform state files in Google Cloud Storage. Format: `gs://{bucket}/{folder}` A default bucket will be bootstrapped if the field is not set or empty')