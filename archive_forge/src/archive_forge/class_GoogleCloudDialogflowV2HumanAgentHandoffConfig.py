from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentHandoffConfig(_messages.Message):
    """Defines the hand off to a live agent, typically on which external agent
  service provider to connect to a conversation. Currently, this feature is
  not general available, please contact Google to get access.

  Fields:
    livePersonConfig: Uses LivePerson (https://www.liveperson.com).
    salesforceLiveAgentConfig: Uses Salesforce Live Agent.
  """
    livePersonConfig = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentHandoffConfigLivePersonConfig', 1)
    salesforceLiveAgentConfig = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentHandoffConfigSalesforceLiveAgentConfig', 2)