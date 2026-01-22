from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1KnowledgeConnectorSettings(_messages.Message):
    """The Knowledge Connector settings for this page or flow. This includes
  information such as the attached Knowledge Bases, and the way to execute
  fulfillment.

  Fields:
    dataStoreConnections: Optional. List of related data store connections.
    enabled: Whether Knowledge Connector is enabled or not.
    targetFlow: The target flow to transition to. Format:
      `projects//locations//agents//flows/`.
    targetPage: The target page to transition to. Format:
      `projects//locations//agents//flows//pages/`.
    triggerFulfillment: The fulfillment to be triggered. When the answers from
      the Knowledge Connector are selected by Dialogflow, you can utitlize the
      request scoped parameter `$request.knowledge.answers` (contains up to
      the 5 highest confidence answers) and `$request.knowledge.questions`
      (contains the corresponding questions) to construct the fulfillment.
  """
    dataStoreConnections = _messages.MessageField('GoogleCloudDialogflowCxV3beta1DataStoreConnection', 1, repeated=True)
    enabled = _messages.BooleanField(2)
    targetFlow = _messages.StringField(3)
    targetPage = _messages.StringField(4)
    triggerFulfillment = _messages.MessageField('GoogleCloudDialogflowCxV3beta1Fulfillment', 5)