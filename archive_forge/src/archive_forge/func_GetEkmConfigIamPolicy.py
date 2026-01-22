from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base
from googlecloudsdk.command_lib.iam import iam_util
def GetEkmConfigIamPolicy(ekm_config_name):
    """Fetch the IAM Policy attached to the EkmConfig.

  Args:
      ekm_config_name: A string name of the EkmConfig.

  Returns:
      An apitools wrapper for the IAM Policy.
  """
    client = base.GetClientInstance()
    messages = base.GetMessagesModule()
    req = messages.CloudkmsProjectsLocationsEkmConfigGetIamPolicyRequest(options_requestedPolicyVersion=iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION, resource=ekm_config_name)
    return client.projects_locations_ekmConfig.GetIamPolicy(req)