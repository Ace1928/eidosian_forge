from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1UpdateSpecialistPoolOperationMetadata(_messages.Message):
    """Runtime operation metadata for
  SpecialistPoolService.UpdateSpecialistPool.

  Fields:
    genericMetadata: The operation generic information.
    specialistPool: Output only. The name of the SpecialistPool to which the
      specialists are being added. Format: `projects/{project_id}/locations/{l
      ocation_id}/specialistPools/{specialist_pool}`
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GenericOperationMetadata', 1)
    specialistPool = _messages.StringField(2)