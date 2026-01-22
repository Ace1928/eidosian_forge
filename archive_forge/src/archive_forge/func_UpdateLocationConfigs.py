from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.immersive_stream.xr import api_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def UpdateLocationConfigs(release_track, instance_ref, target_location_configs):
    """Updates the location configs for a service instance.

  Args:
    release_track: ALPHA or GA release track
    instance_ref: resource object - service instance to be updated
    target_location_configs: A LocationConfigsValue proto message represents the
      target location configs to achieve

  Returns:
    An Operation object which can be used to check on the progress of the
    service instance update.
  """
    if not target_location_configs or not target_location_configs.additionalProperties:
        raise exceptions.Error('Target location configs must be provided')
    client = api_util.GetClient(release_track)
    messages = api_util.GetMessages(release_track)
    instance = messages.StreamInstance(locationConfigs=target_location_configs)
    service = client.ProjectsLocationsStreamInstancesService(client)
    return service.Patch(messages.StreamProjectsLocationsStreamInstancesPatchRequest(name=instance_ref.RelativeName(), streamInstance=instance, updateMask='location_configs'))