from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.command_lib.compute.os_config.troubleshoot import utils
from googlecloudsdk.command_lib.projects import util
def _FailEnablementMessage(project_id):
    service_account = 'service-{}@gcp-sa-osconfig.iam.gserviceaccount.com'.format(util.GetProjectNumber(project_id))
    return 'No\nNo OS Config service account is present and enabled for this instance. To create an OS Config service account for this instance, visit https://cloud.google.com/compute/docs/access/create-enable-service-accounts-for-instances#createanewserviceaccount to create a service account of the name ' + service_account + ', grant it the "Cloud OS Config Service Agent" IAM role, then disable and re-enable the OS Config API.'