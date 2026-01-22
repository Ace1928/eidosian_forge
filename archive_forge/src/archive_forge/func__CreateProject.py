from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.api_lib.resource_manager import operations
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.command_lib.projects import util as  projects_command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def _CreateProject(project_id, project_ids):
    """Create a project and check that it isn't in the known project IDs."""
    if project_ids and project_id in project_ids:
        raise ValueError('Attempting to create a project that already exists.')
    project_ref = resources.REGISTRY.Create('cloudresourcemanager.projects', projectId=project_id)
    try:
        create_op = projects_api.Create(project_ref)
    except Exception as err:
        log.warning('Project creation failed: {err}\nPlease make sure to create the project [{project}] using\n    $ gcloud projects create {project}\nor change to another project using\n    $ gcloud config set project <PROJECT ID>'.format(err=six.text_type(err), project=project_id))
        return None
    try:
        create_op = operations.WaitForOperation(create_op)
    except operations.OperationFailedException as err:
        log.warning('Project creation for project [{project}] failed:\n  {err}'.format(err=six.text_type(err), project=project_id))
        return None
    return project_id