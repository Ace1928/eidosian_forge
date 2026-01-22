from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.command_lib.compute.os_config.troubleshoot import utils
from googlecloudsdk.command_lib.projects import util
def CheckEnablement(project):
    """Checks whether there is an enabled OS Config service account."""
    response_message = '> Is the OS Config Service account present for this instance? '
    continue_flag = False
    iam_policy = None
    project_ref = util.ParseProject(project.name)
    try:
        iam_policy = projects_api.GetIamPolicy(project_ref)
    except exceptions.HttpError as e:
        response_message += utils.UnknownMessage(e)
        return utils.Response(continue_flag, response_message)
    for binding in iam_policy.bindings:
        if binding.role == 'roles/osconfig.serviceAgent':
            if not binding.members:
                break
            else:
                project_number = str(util.GetProjectNumber(project.name))
                for member in binding.members:
                    if project_number in member:
                        response_message += 'Yes'
                        continue_flag = True
                        return utils.Response(continue_flag, response_message)
                service_account = 'service-{}@gcp-sa-osconfig.iam.gserviceaccount.com'.format(project_number)
                response_message += 'Yes\nHowever, the service account name does not contain a matching project number. The service account should be of the name ' + service_account
                return utils.Response(continue_flag, response_message)
    response_message += _FailEnablementMessage(project.name)
    return utils.Response(continue_flag, response_message)