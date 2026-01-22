from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def BackendBucketArgument(plural=False):
    return compute_flags.ResourceArgument(name='backend_bucket_name', resource_name='backend bucket', plural=plural, completer=BackendBucketsCompleter, global_collection='compute.backendBuckets')