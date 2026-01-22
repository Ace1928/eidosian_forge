from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputs(_messages.Message):
    """A
  GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputs
  object.

  Fields:
    additionalExperiments: Additional experiment flags for the Tables training
      pipeline.
    disableEarlyStopping: Use the entire training budget. This disables the
      early stopping feature. By default, the early stopping feature is
      enabled, which means that AutoML Tables might stop training before the
      entire training budget has been used.
    exportEvaluatedDataItemsConfig: Configuration for exporting test set
      predictions to a BigQuery table. If this configuration is absent, then
      the export is not performed.
    optimizationObjective: Objective function the model is optimizing towards.
      The training process creates a model that maximizes/minimizes the value
      of the objective function over the validation set. The supported
      optimization objectives depend on the prediction type. If the field is
      not set, a default objective function is used. classification (binary):
      "maximize-au-roc" (default) - Maximize the area under the receiver
      operating characteristic (ROC) curve. "minimize-log-loss" - Minimize log
      loss. "maximize-au-prc" - Maximize the area under the precision-recall
      curve. "maximize-precision-at-recall" - Maximize precision for a
      specified recall value. "maximize-recall-at-precision" - Maximize recall
      for a specified precision value. classification (multi-class):
      "minimize-log-loss" (default) - Minimize log loss. regression:
      "minimize-rmse" (default) - Minimize root-mean-squared error (RMSE).
      "minimize-mae" - Minimize mean-absolute error (MAE). "minimize-rmsle" -
      Minimize root-mean-squared log error (RMSLE).
    optimizationObjectivePrecisionValue: Required when optimization_objective
      is "maximize-recall-at-precision". Must be between 0 and 1, inclusive.
    optimizationObjectiveRecallValue: Required when optimization_objective is
      "maximize-precision-at-recall". Must be between 0 and 1, inclusive.
    predictionType: The type of prediction the Model is to produce.
      "classification" - Predict one out of multiple target values is picked
      for each row. "regression" - Predict a value based on its relation to
      other values. This type is available only to columns that contain
      semantically numeric values, i.e. integers or floating point number,
      even if stored as e.g. strings.
    targetColumn: The column name of the target column that the model is to
      predict.
    trainBudgetMilliNodeHours: Required. The train budget of creating this
      model, expressed in milli node hours i.e. 1,000 value in this field
      means 1 node hour. The training cost of the model will not exceed this
      budget. The final cost will be attempted to be close to the budget,
      though may end up being (even) noticeably smaller - at the backend's
      discretion. This especially may happen when further model training
      ceases to provide any improvements. If the budget is set to a value
      known to be insufficient to train a model for the given dataset, the
      training won't be attempted and will error. The train budget must be
      between 1,000 and 72,000 milli node hours, inclusive.
    transformations: Each transformation will apply transform function to
      given input column. And the result will be used for training. When
      creating transformation for BigQuery Struct column, the column should be
      flattened using "." as the delimiter.
    weightColumnName: Column name that should be used as the weight column.
      Higher values in this column give more importance to the row during
      model training. The column must have numeric values between 0 and 10000
      inclusively; 0 means the row is ignored for training. If weight column
      field is not set, then all rows are assumed to have equal weight of 1.
  """
    additionalExperiments = _messages.StringField(1, repeated=True)
    disableEarlyStopping = _messages.BooleanField(2)
    exportEvaluatedDataItemsConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionExportEvaluatedDataItemsConfig', 3)
    optimizationObjective = _messages.StringField(4)
    optimizationObjectivePrecisionValue = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    optimizationObjectiveRecallValue = _messages.FloatField(6, variant=_messages.Variant.FLOAT)
    predictionType = _messages.StringField(7)
    targetColumn = _messages.StringField(8)
    trainBudgetMilliNodeHours = _messages.IntegerField(9)
    transformations = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformation', 10, repeated=True)
    weightColumnName = _messages.StringField(11)