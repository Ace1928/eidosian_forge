from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlVideoObjectTrackingInputs(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlVideoObjectTra
  ckingInputs object.

  Enums:
    ModelTypeValueValuesEnum:

  Fields:
    modelType: A ModelTypeValueValuesEnum attribute.
  """

    class ModelTypeValueValuesEnum(_messages.Enum):
        """ModelTypeValueValuesEnum enum type.

    Values:
      MODEL_TYPE_UNSPECIFIED: Should not be set.
      CLOUD: A model best tailored to be used within Google Cloud, and which c
        annot be exported. Default.
      MOBILE_VERSATILE_1: A model that, in addition to being available within
        Google Cloud, can also be exported (see ModelService.ExportModel) as a
        TensorFlow or TensorFlow Lite model and used on a mobile or edge
        device afterwards.
      MOBILE_CORAL_VERSATILE_1: A versatile model that is meant to be exported
        (see ModelService.ExportModel) and used on a Google Coral device.
      MOBILE_CORAL_LOW_LATENCY_1: A model that trades off quality for low
        latency, to be exported (see ModelService.ExportModel) and used on a
        Google Coral device.
      MOBILE_JETSON_VERSATILE_1: A versatile model that is meant to be
        exported (see ModelService.ExportModel) and used on an NVIDIA Jetson
        device.
      MOBILE_JETSON_LOW_LATENCY_1: A model that trades off quality for low
        latency, to be exported (see ModelService.ExportModel) and used on an
        NVIDIA Jetson device.
    """
        MODEL_TYPE_UNSPECIFIED = 0
        CLOUD = 1
        MOBILE_VERSATILE_1 = 2
        MOBILE_CORAL_VERSATILE_1 = 3
        MOBILE_CORAL_LOW_LATENCY_1 = 4
        MOBILE_JETSON_VERSATILE_1 = 5
        MOBILE_JETSON_LOW_LATENCY_1 = 6
    modelType = _messages.EnumField('ModelTypeValueValuesEnum', 1)