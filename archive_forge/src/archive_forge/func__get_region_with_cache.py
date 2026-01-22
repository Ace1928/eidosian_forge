from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import textwrap
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import sign_url_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
@functools.lru_cache(maxsize=None)
def _get_region_with_cache(scheme, bucket_name):
    api_client = api_factory.get_api(scheme)
    try:
        bucket_resource = api_client.get_bucket(bucket_name)
    except api_errors.CloudApiError:
        raise command_errors.Error("Failed to auto-detect the region for {}. Please ensure you have storage.buckets.get permission on the bucket, or specify the bucket's region using the '--region' flag.".format(bucket_name))
    return bucket_resource.location