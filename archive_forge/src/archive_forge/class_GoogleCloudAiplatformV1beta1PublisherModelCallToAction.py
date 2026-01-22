from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PublisherModelCallToAction(_messages.Message):
    """Actions could take on this Publisher Model.

  Fields:
    createApplication: Optional. Create application using the PublisherModel.
    deploy: Optional. Deploy the PublisherModel to Vertex Endpoint.
    deployGke: Optional. Deploy PublisherModel to Google Kubernetes Engine.
    openEvaluationPipeline: Optional. Open evaluation pipeline of the
      PublisherModel.
    openFineTuningPipeline: Optional. Open fine-tuning pipeline of the
      PublisherModel.
    openFineTuningPipelines: Optional. Open fine-tuning pipelines of the
      PublisherModel.
    openGenerationAiStudio: Optional. Open in Generation AI Studio.
    openGenie: Optional. Open Genie / Playground.
    openNotebook: Optional. Open notebook of the PublisherModel.
    openNotebooks: Optional. Open notebooks of the PublisherModel.
    openPromptTuningPipeline: Optional. Open prompt-tuning pipeline of the
      PublisherModel.
    requestAccess: Optional. Request for access.
    viewRestApi: Optional. To view Rest API docs.
  """
    createApplication = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionRegionalResourceReferences', 1)
    deploy = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionDeploy', 2)
    deployGke = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionDeployGke', 3)
    openEvaluationPipeline = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionRegionalResourceReferences', 4)
    openFineTuningPipeline = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionRegionalResourceReferences', 5)
    openFineTuningPipelines = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionOpenFineTuningPipelines', 6)
    openGenerationAiStudio = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionRegionalResourceReferences', 7)
    openGenie = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionRegionalResourceReferences', 8)
    openNotebook = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionRegionalResourceReferences', 9)
    openNotebooks = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionOpenNotebooks', 10)
    openPromptTuningPipeline = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionRegionalResourceReferences', 11)
    requestAccess = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionRegionalResourceReferences', 12)
    viewRestApi = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionViewRestApi', 13)