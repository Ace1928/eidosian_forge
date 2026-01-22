from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import automation
from googlecloudsdk.command_lib.deploy import exceptions as cd_exceptions
from googlecloudsdk.core import resources
def AutomationReference(automation_name, project, location_id):
    """Creates the automation reference base on the parameters.

  Returns the reference of the automation name.

  Args:
    automation_name: str, in the format of pipeline_id/automation_id.
    project: str,project number or ID.
    location_id: str, region ID.

  Returns:
    Automation name reference.
  """
    try:
        pipeline_id, automation_id = automation_name.split('/')
    except ValueError:
        raise cd_exceptions.AutomationNameFormatError(automation_name)
    return resources.REGISTRY.Parse(None, collection=_AUTOMATION_COLLECTION, params={'projectsId': project, 'locationsId': location_id, 'deliveryPipelinesId': pipeline_id, 'automationsId': automation_id})