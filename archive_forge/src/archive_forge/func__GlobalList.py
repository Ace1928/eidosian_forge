from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.run import platforms
from surface.run.services import list as services_list
def _GlobalList(self, client):
    """Provides the method to provide a regionless list."""
    return global_methods.ListServices(client, field_selector='multiRegionOnly')