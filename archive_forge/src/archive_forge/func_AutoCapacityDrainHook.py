from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AutoCapacityDrainHook(api_version='v1'):
    """Hook to transform AutoCapacityDrain flag to actual message.

  This function is called during ServiceLbPolicy create/update command to
  create the AutoCapacityDrain message. It returns a function which is called
  with arguments passed in the gcloud command.

  Args:
    api_version: Version of the networkservices api

  Returns:
     Function to transform boolean flag to AutcapacityDrain message.
  """
    messages = apis.GetMessagesModule('networkservices', api_version)

    def ConstructAutoCapacityDrain(enable):
        if enable:
            return messages.ServiceLbPolicyAutoCapacityDrain(enable=enable)
    return ConstructAutoCapacityDrain