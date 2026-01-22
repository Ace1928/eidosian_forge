from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
def api_get(name):
    """Returns job details from API as Apitools object."""
    client = apis.GetClientInstance('transfer', 'v1')
    messages = apis.GetMessagesModule('transfer', 'v1')
    formatted_job_name = name_util.add_job_prefix(name)
    return client.transferJobs.Get(messages.StoragetransferTransferJobsGetRequest(jobName=formatted_job_name, projectId=properties.VALUES.core.project.Get()))