from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutomlImageTrainingTunableParameter(_messages.Message):
    """A wrapper class which contains the tunable parameters in an AutoML Image
  training job.

  Enums:
    TrainerTypeValueValuesEnum:

  Messages:
    DatasetConfigValue: Customizable dataset settings, used in the
      `model_garden_trainer`.
    TrainerConfigValue: Customizable trainer settings, used in the
      `model_garden_trainer`.

  Fields:
    checkpointName: Optional. An unique name of pretrained model checkpoint
      provided in model garden, it will be mapped to a GCS location
      internally.
    datasetConfig: Customizable dataset settings, used in the
      `model_garden_trainer`.
    studySpec: Optioinal. StudySpec of hyperparameter tuning job. Required for
      `model_garden_trainer`.
    trainerConfig: Customizable trainer settings, used in the
      `model_garden_trainer`.
    trainerType: A TrainerTypeValueValuesEnum attribute.
  """

    class TrainerTypeValueValuesEnum(_messages.Enum):
        """TrainerTypeValueValuesEnum enum type.

    Values:
      TRAINER_TYPE_UNSPECIFIED: Default value.
      AUTOML_TRAINER: <no description>
      MODEL_GARDEN_TRAINER: <no description>
    """
        TRAINER_TYPE_UNSPECIFIED = 0
        AUTOML_TRAINER = 1
        MODEL_GARDEN_TRAINER = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DatasetConfigValue(_messages.Message):
        """Customizable dataset settings, used in the `model_garden_trainer`.

    Messages:
      AdditionalProperty: An additional property for a DatasetConfigValue
        object.

    Fields:
      additionalProperties: Additional properties of type DatasetConfigValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DatasetConfigValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TrainerConfigValue(_messages.Message):
        """Customizable trainer settings, used in the `model_garden_trainer`.

    Messages:
      AdditionalProperty: An additional property for a TrainerConfigValue
        object.

    Fields:
      additionalProperties: Additional properties of type TrainerConfigValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TrainerConfigValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    checkpointName = _messages.StringField(1)
    datasetConfig = _messages.MessageField('DatasetConfigValue', 2)
    studySpec = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpec', 3)
    trainerConfig = _messages.MessageField('TrainerConfigValue', 4)
    trainerType = _messages.EnumField('TrainerTypeValueValuesEnum', 5)