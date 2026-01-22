from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
class ImageVersionService(object):
    """Provides supported images version from the Image Version API."""

    def __init__(self, release_track=base.ReleaseTrack.GA):
        self.client = None
        self.release_track = release_track
        self.messages = api_util.GetMessagesModule(release_track=self.release_track)

    def GetClient(self):
        if self.client is None:
            self.client = api_util.GetClientInstance(self.release_track).projects_locations_imageVersions
        return self.client

    def List(self, project_location_ref):
        """Retrieves list of supported images version from the Image Version API."""
        request = self.messages.ComposerProjectsLocationsImageVersionsListRequest
        locations = [project_location_ref]
        return api_util.AggregateListResults(request, self.GetClient(), locations, LIST_FIELD_NAME, PAGE_SIZE)