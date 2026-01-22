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
def AnyFlexibleCacheArgsSpecified(self, args):
    """Returns true if any Flexible Cache args for updating backend bucket were specified."""
    return any((args.IsSpecified('cache_mode'), args.IsSpecified('client_ttl'), args.IsSpecified('no_client_ttl'), args.IsSpecified('default_ttl'), args.IsSpecified('no_default_ttl'), args.IsSpecified('max_ttl'), args.IsSpecified('no_max_ttl'), args.IsSpecified('custom_response_header'), args.IsSpecified('no_custom_response_headers'), args.IsSpecified('negative_caching'), args.IsSpecified('negative_caching_policy'), args.IsSpecified('no_negative_caching_policies'), args.IsSpecified('serve_while_stale'), args.IsSpecified('no_serve_while_stale'), args.IsSpecified('bypass_cache_on_request_headers'), args.IsSpecified('no_bypass_cache_on_request_headers')))