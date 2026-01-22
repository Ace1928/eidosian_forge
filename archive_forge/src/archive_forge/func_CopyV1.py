from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
def CopyV1(self, destination_region_ref=None, source_model=None, kms_key_name=None, destination_model_id=None, destination_parent_model=None):
    """Copies the given source model into specified location.

    The source model is copied into specified location (including cross-region)
    either as a new model or a new model version under given parent model.

    Args:
      destination_region_ref: the resource reference to the location into which
        to copy the Model.
      source_model: The resource name of the Model to copy.
      kms_key_name: The name of the KMS key to use for model encryption.
      destination_model_id: Optional. Thew custom ID to be used as the resource
        name of the new model. This value may be up to 63 characters, and valid
        characters are  `[a-z0-9_-]`. The first character cannot be a number or
        hyphen.
      destination_parent_model: The destination parent model to copy the model
        as a model version into.

    Returns:
      Response from calling copy model.
    """
    encryption_spec = None
    if kms_key_name:
        encryption_spec = self.messages.GoogleCloudAiplatformV1EncryptionSpec(kmsKeyName=kms_key_name)
    request = self.messages.AiplatformProjectsLocationsModelsCopyRequest(parent=destination_region_ref.RelativeName(), googleCloudAiplatformV1CopyModelRequest=self.messages.GoogleCloudAiplatformV1CopyModelRequest(sourceModel=source_model, encryptionSpec=encryption_spec, parentModel=destination_parent_model, modelId=destination_model_id))
    return self._service.Copy(request)