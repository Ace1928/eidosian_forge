from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchMigrateResourcesOperationMetadata(_messages.Message):
    """Runtime operation information for
  MigrationService.BatchMigrateResources.

  Fields:
    genericMetadata: The common part of the operation metadata.
    partialResults: Partial results that reflect the latest migration
      operation progress.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GenericOperationMetadata', 1)
    partialResults = _messages.MessageField('GoogleCloudAiplatformV1beta1BatchMigrateResourcesOperationMetadataPartialResult', 2, repeated=True)