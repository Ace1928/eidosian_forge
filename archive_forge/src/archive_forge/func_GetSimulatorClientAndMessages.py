from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def GetSimulatorClientAndMessages(api_version):
    client = apis.GetClientInstance(_API_NAME, api_version)
    messages = apis.GetMessagesModule(_API_NAME, api_version)
    return (client, messages)