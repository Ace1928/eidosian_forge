from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlImageClassificationInputs(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlImageClassific
  ationInputs object.

  Enums:
    ModelTypeValueValuesEnum:

  Fields:
    baseModelId: The ID of the `base` model. If it is specified, the new model
      will be trained based on the `base` model. Otherwise, the new model will
      be trained from scratch. The `base` model must be in the same Project
      and Location as the new Model to train, and have the same modelType.
    budgetMilliNodeHours: The training budget of creating this model,
      expressed in milli node hours i.e. 1,000 value in this field means 1
      node hour. The actual metadata.costMilliNodeHours will be equal or less
      than this value. If further model training ceases to provide any
      improvements, it will stop without using the full budget and the
      metadata.successfulStopReason will be `model-converged`. Note, node_hour
      = actual_hour * number_of_nodes_involved. For modelType
      `cloud`(default), the budget must be between 8,000 and 800,000 milli
      node hours, inclusive. The default value is 192,000 which represents one
      day in wall time, considering 8 nodes are used. For model types `mobile-
      tf-low-latency-1`, `mobile-tf-versatile-1`, `mobile-tf-high-accuracy-1`,
      the training budget must be between 1,000 and 100,000 milli node hours,
      inclusive. The default value is 24,000 which represents one day in wall
      time on a single node that is used.
    disableEarlyStopping: Use the entire training budget. This disables the
      early stopping feature. When false the early stopping feature is
      enabled, which means that AutoML Image Classification might stop
      training before the entire training budget has been used.
    modelType: A ModelTypeValueValuesEnum attribute.
    multiLabel: If false, a single-label (multi-class) Model will be trained
      (i.e. assuming that for each image just up to one annotation may be
      applicable). If true, a multi-label Model will be trained (i.e. assuming
      that for each image multiple annotations may be applicable).
    uptrainBaseModelId: The ID of `base` model for upTraining. If it is
      specified, the new model will be upTrained based on the `base` model for
      upTraining. Otherwise, the new model will be trained from scratch. The
      `base` model for upTraining must be in the same Project and Location as
      the new Model to train, and have the same modelType.
  """

    class ModelTypeValueValuesEnum(_messages.Enum):
        """ModelTypeValueValuesEnum enum type.

    Values:
      MODEL_TYPE_UNSPECIFIED: Should not be set.
      CLOUD: A Model best tailored to be used within Google Cloud, and which
        cannot be exported. Default.
      CLOUD_1: A model type best tailored to be used within Google Cloud,
        which cannot be exported externally. Compared to the CLOUD model
        above, it is expected to have higher prediction accuracy.
      MOBILE_TF_LOW_LATENCY_1: A model that, in addition to being available
        within Google Cloud, can also be exported (see
        ModelService.ExportModel) as TensorFlow or Core ML model and used on a
        mobile or edge device afterwards. Expected to have low latency, but
        may have lower prediction quality than other mobile models.
      MOBILE_TF_VERSATILE_1: A model that, in addition to being available
        within Google Cloud, can also be exported (see
        ModelService.ExportModel) as TensorFlow or Core ML model and used on a
        mobile or edge device with afterwards.
      MOBILE_TF_HIGH_ACCURACY_1: A model that, in addition to being available
        within Google Cloud, can also be exported (see
        ModelService.ExportModel) as TensorFlow or Core ML model and used on a
        mobile or edge device afterwards. Expected to have a higher latency,
        but should also have a higher prediction quality than other mobile
        models.
    """
        MODEL_TYPE_UNSPECIFIED = 0
        CLOUD = 1
        CLOUD_1 = 2
        MOBILE_TF_LOW_LATENCY_1 = 3
        MOBILE_TF_VERSATILE_1 = 4
        MOBILE_TF_HIGH_ACCURACY_1 = 5
    baseModelId = _messages.StringField(1)
    budgetMilliNodeHours = _messages.IntegerField(2)
    disableEarlyStopping = _messages.BooleanField(3)
    modelType = _messages.EnumField('ModelTypeValueValuesEnum', 4)
    multiLabel = _messages.BooleanField(5)
    uptrainBaseModelId = _messages.StringField(6)