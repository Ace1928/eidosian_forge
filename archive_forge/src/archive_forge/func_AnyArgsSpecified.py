from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import cdn_flags_utils as cdn_flags
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import signed_url_flags
from googlecloudsdk.command_lib.compute.backend_buckets import backend_buckets_utils
from googlecloudsdk.command_lib.compute.backend_buckets import flags as backend_buckets_flags
from googlecloudsdk.command_lib.compute.security_policies import (
from googlecloudsdk.core import log
def AnyArgsSpecified(self, args):
    """Returns true if any args for updating backend bucket were specified."""
    return args.IsSpecified('description') or args.IsSpecified('gcs_bucket_name') or args.IsSpecified('enable_cdn') or args.IsSpecified('edge_security_policy') or args.IsSpecified('cache_key_include_http_header') or args.IsSpecified('cache_key_query_string_whitelist') or args.IsSpecified('compression_mode')