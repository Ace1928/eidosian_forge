from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.immersive_stream.xr import api_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def UpdateContentBuildVersion(release_track, instance_ref, version):
    """Update content build version of an Immersive Stream for XR service instance.

  Args:
    release_track: ALPHA or GA release track
    instance_ref: resource object - service instance to be updated
    version: content build version tag

  Returns:
    An Operation object which can be used to check on the progress of the
    service instance update.
  """
    client = api_util.GetClient(release_track)
    messages = api_util.GetMessages(release_track)
    build_version = messages.BuildVersion(contentVersionTag=version)
    instance = messages.StreamInstance(contentBuildVersion=build_version)
    service = client.ProjectsLocationsStreamInstancesService(client)
    return service.Patch(messages.StreamProjectsLocationsStreamInstancesPatchRequest(name=instance_ref.RelativeName(), streamInstance=instance, updateMask='content_build_version'))