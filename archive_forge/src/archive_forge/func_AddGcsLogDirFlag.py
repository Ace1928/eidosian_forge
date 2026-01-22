from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def AddGcsLogDirFlag(parser, hidden=False):
    """Add a GCS directory flag to hold build logs."""
    parser.add_argument('--gcs-log-dir', hidden=hidden, help='A directory in Google Cloud Storage to hold build logs. If this field is not set, ```gs://[PROJECT_NUMBER].cloudbuild-logs.googleusercontent.com/``` will be created and used or ```gs://[PROJECT_NUMBER]-[builds/region]-cloudbuild-logs``` is used when you set `--default-buckets-behavior` to `REGIONAL_USER_OWNED_BUCKET`.')