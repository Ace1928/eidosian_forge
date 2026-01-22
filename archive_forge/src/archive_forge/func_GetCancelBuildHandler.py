from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.core import log
from googlecloudsdk.core.util import keyboard_interrupt
import six
def GetCancelBuildHandler(client, messages, build_ref):
    """Returns a handler to cancel a build.

  Args:
    client: base_api.BaseApiClient, An instance of the Cloud Build client.
    messages: Module containing the definitions of messages for Cloud Build.
    build_ref: Build reference. Expects a cloudbuild.projects.locations.builds
      but also supports cloudbuild.projects.builds.
  """

    def _CancelBuildHandler(unused_signal_number, unused_stack_frame):
        """Cancels the build_ref build.

    Args:
      unused_signal_number: The signal caught.
      unused_stack_frame: The interrupt stack frame.

    Raises:
      InvalidUserInputError: if project ID or build ID is not specified.
    """
        log.status.Print('Cancelling...')
        project_id = None
        if hasattr(build_ref, 'projectId'):
            project_id = build_ref.projectId
        elif hasattr(build_ref, 'projectsId'):
            project_id = build_ref.projectsId
        build_id = None
        if hasattr(build_ref, 'id'):
            build_id = build_ref.id
        elif hasattr(build_ref, 'buildsId'):
            build_id = build_ref.buildsId
        location = None
        if hasattr(build_ref, 'locationsId'):
            location = build_ref.locationsId
        if location is not None:
            cancel_name = 'projects/{project}/locations/{location}/builds/{buildId}'
            name = cancel_name.format(project=project_id, location=location, buildId=build_id)
            client.projects_locations_builds.Cancel(messages.CancelBuildRequest(name=name))
        else:
            client.projects_builds.Cancel(messages.CloudbuildProjectsBuildsCancelRequest(projectId=project_id, id=build_id))
        log.status.Print('Cancelled [{r}].'.format(r=six.text_type(build_ref)))
    return _CancelBuildHandler