from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def ServiceExists(args, project, service_name, region, release_track):
    """Check to see if the service with the given name already exists."""
    context = connection_context.GetConnectionContext(args, release_track=release_track, platform=platforms.PLATFORM_MANAGED, region_label=region)
    with serverless_operations.Connect(context) as client:
        return client.GetService(_ServiceResource(project, service_name))