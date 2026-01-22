from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ExplanationInput(_messages.Message):
    """Represents input parameters for a model explanation job.

  Enums:
    DataFormatValueValuesEnum: Required. The format of the input data.
    FrameworkValueValuesEnum: Optional. The framework used to train this
      model. Only needed if model_version is a GCS path. Otherwise the
      framework specified during version creation will be used.
    OutputDataFormatValueValuesEnum: Optional. The format of the output data,
      defaults to BIGQUERY.

  Fields:
    accelerator: Optional. The type and number of accelerators to be attached
      to each machine running the job.
    batchSize: Optional. Number of records per batch, defaults to 64. The
      service will buffer batch_size number of records in memory before
      invoking one Tensorflow prediction call internally. So take the record
      size and memory available into consideration when setting this
      parameter.
    dataFormat: Required. The format of the input data.
    explanationConfig: Required only if model_version is specified through a
      uri, otherwise the same explanation config specified at model version
      creation will be used. Configures explainability features on the model's
      version. Some explanation features require additional metadata to be
      loaded as part of the model payload.
    framework: Optional. The framework used to train this model. Only needed
      if model_version is a GCS path. Otherwise the framework specified during
      version creation will be used.
    initialWorkerCount: Optional. The initial number of workers to be used for
      parallel processing. Defaults to 0 if one wants the service to figure
      out the number. The actual number of workers being used may change after
      the job starts depending on the autoscaling policy.
    inputPaths: Required when data_format is JSON. The Cloud Storage location
      of the input data. May contain wildcards.
    maxWorkerCount: Optional. The maximum number of workers to be used for
      parallel processing. Defaults to 10 if not specified.
    modelName: Use this field if you want to use the default version for the
      specified model. The string must use the following format:
      `"projects/YOUR_PROJECT/models/YOUR_MODEL"`
    outputBigqueryTable: Required when output_data_format is BIGQUERY. The
      output fully qualified BigQuery table name in the format of
      "[project_id].[dataset_name].[table_name]".
    outputDataFormat: Optional. The format of the output data, defaults to
      BIGQUERY.
    region: Required. The Compute Engine region to run the explanation job in.
      See the available regions for AI Platform services.
    runtimeVersion: Required. The AI Platform runtime version to use for the
      explanation job. See <a href="https://cloud.google.com/ml-
      engine/docs/tensorflow/runtime-version-list</a> for available runtime
      versions. Must be >=1.12.
    signatureName: Optional. The name of the signature defined in the
      SavedModel to use for this job. Please refer to
      [SavedModel](https://tensorflow.github.io/serving/serving_basic.html)
      for information about how to use signatures. Defaults to [DEFAULT_SERVIN
      G_SIGNATURE_DEF_KEY](https://www.tensorflow.org/api_docs/python/tf/saved
      _model/signature_constants) , which is "serving_default".
    tagsOverride: Optional. The set of tags to select which meta graph defined
      in the SavedModel to use for this job. Please refer to
      [SavedModel](https://www.tensorflow.org/serving/serving_basic) for
      information about how to use tags. Overrides the default tags when
      predicting from a deployed model version. When predicting from a model
      directory, the tag defaults to [SERVING](https://www.tensorflow.org/api_
      docs/python/tf/saved_model/tag_constants) , which is "serve".
    uri: Use this field if you want to specify a Google Cloud Storage path for
      the model to use, e.g. gs://{BUCKET}/{MODEL_DIR}/{MODEL_NAME}.
    versionName: Use this field if you want to specify a version of the model
      to use. The string is formatted the same way as `model_version`, with
      the addition of the version information:
      `"projects/YOUR_PROJECT/models/YOUR_MODEL/versions/YOUR_VERSION"`
    workerType: Optional. The type of virtual machine to use for the
      explanation job's worker nodes. It supports all machine types available
      on GCP ( https://cloud.google.com/compute/docs/machine-types), subject
      to the availability in the specific region the job runs.
  """

    class DataFormatValueValuesEnum(_messages.Enum):
        """Required. The format of the input data.

    Values:
      DATA_FORMAT_UNSPECIFIED: Unspecified format.
      JSON: Each line of the file is a JSON dictionary representing one
        record. Currently available only for input data.
      BIGQUERY: Values are rows in a BigQuery table given its associated
        schema. Currently available only for output data.
    """
        DATA_FORMAT_UNSPECIFIED = 0
        JSON = 1
        BIGQUERY = 2

    class FrameworkValueValuesEnum(_messages.Enum):
        """Optional. The framework used to train this model. Only needed if
    model_version is a GCS path. Otherwise the framework specified during
    version creation will be used.

    Values:
      FRAMEWORK_UNSPECIFIED: Unspecified framework. Assigns a value based on
        the file suffix.
      TENSORFLOW: Tensorflow framework.
      SCIKIT_LEARN: Scikit-learn framework.
      XGBOOST: XGBoost framework.
    """
        FRAMEWORK_UNSPECIFIED = 0
        TENSORFLOW = 1
        SCIKIT_LEARN = 2
        XGBOOST = 3

    class OutputDataFormatValueValuesEnum(_messages.Enum):
        """Optional. The format of the output data, defaults to BIGQUERY.

    Values:
      DATA_FORMAT_UNSPECIFIED: Unspecified format.
      JSON: Each line of the file is a JSON dictionary representing one
        record. Currently available only for input data.
      BIGQUERY: Values are rows in a BigQuery table given its associated
        schema. Currently available only for output data.
    """
        DATA_FORMAT_UNSPECIFIED = 0
        JSON = 1
        BIGQUERY = 2
    accelerator = _messages.MessageField('GoogleCloudMlV1AcceleratorConfig', 1)
    batchSize = _messages.IntegerField(2)
    dataFormat = _messages.EnumField('DataFormatValueValuesEnum', 3)
    explanationConfig = _messages.MessageField('GoogleCloudMlV1ExplanationConfig', 4)
    framework = _messages.EnumField('FrameworkValueValuesEnum', 5)
    initialWorkerCount = _messages.IntegerField(6)
    inputPaths = _messages.StringField(7, repeated=True)
    maxWorkerCount = _messages.IntegerField(8)
    modelName = _messages.StringField(9)
    outputBigqueryTable = _messages.StringField(10)
    outputDataFormat = _messages.EnumField('OutputDataFormatValueValuesEnum', 11)
    region = _messages.StringField(12)
    runtimeVersion = _messages.StringField(13)
    signatureName = _messages.StringField(14)
    tagsOverride = _messages.StringField(15, repeated=True)
    uri = _messages.StringField(16)
    versionName = _messages.StringField(17)
    workerType = _messages.StringField(18)