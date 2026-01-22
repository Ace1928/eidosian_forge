from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def PromptToEnableApiIfDisabled(service_name, enable_by_default=False):
    """Prompts to enable the API if it's not enabled.

  Args:
    service_name: The name of the service to enable.
    enable_by_default: default choice for the enablement prompt.
  """
    project_id = properties.VALUES.core.project.GetOrFail()
    try:
        if enable_api.IsServiceEnabled(project_id, service_name):
            return
        if console_io.CanPrompt():
            api_enablement.PromptToEnableApi(project_id, service_name, enable_by_default=enable_by_default)
        else:
            log.warning('Service {} is not enabled. This operation may not succeed.'.format(service_name))
    except exceptions.GetServicePermissionDeniedException:
        log.info("Could not verify if service {} is enabled: missing permission 'serviceusage.services.get'.".format(service_name))