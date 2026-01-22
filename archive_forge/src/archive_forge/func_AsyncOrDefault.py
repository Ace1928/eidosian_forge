from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core.console import progress_tracker
def AsyncOrDefault(async_):
    if async_ is None:
        return platforms.GetPlatform() != platforms.PLATFORM_MANAGED
    return async_