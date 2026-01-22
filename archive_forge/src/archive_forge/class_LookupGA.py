from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.apphub import service_projects as apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.GA)
class LookupGA(base.DescribeCommand):
    """Lookup an Apphub service project."""
    detailed_help = _DETAILED_HELP

    def Run(self, args):
        """Run the lookup command."""
        client = apis.ServiceProjectsClient(release_track=base.ReleaseTrack.GA)
        service_project = properties.VALUES.core.project.Get()
        return client.Lookup(service_project=service_project)