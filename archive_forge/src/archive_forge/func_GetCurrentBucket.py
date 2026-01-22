from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def GetCurrentBucket(self, args):
    """Returns a bucket specified by the arguments.

    Loads the current bucket at most once. If called multiple times, the
    previously-loaded bucket will be returned.

    Args:
      args: The argument set. This is not checked across GetCurrentBucket calls,
        and must be consistent.
    """
    if not self._current_bucket:
        return util.GetClient().projects_locations_buckets.Get(util.GetMessages().LoggingProjectsLocationsBucketsGetRequest(name=util.CreateResourceName(util.CreateResourceName(util.GetProjectResource(args.project).RelativeName(), 'locations', args.location), 'buckets', args.BUCKET_ID)))
    return self._current_bucket