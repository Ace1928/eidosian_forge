from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrderedJob(_messages.Message):
    """A job executed by the workflow.

  Messages:
    LabelsValue: Optional. The labels to associate with this job.Label keys
      must be between 1 and 63 characters long, and must conform to the
      following regular expression: \\p{Ll}\\p{Lo}{0,62}Label values must be
      between 1 and 63 characters long, and must conform to the following
      regular expression: \\p{Ll}\\p{Lo}\\p{N}_-{0,63}No more than 32 labels can
      be associated with a given job.

  Fields:
    flinkJob: Optional. Job is a Flink job.
    hadoopJob: Optional. Job is a Hadoop job.
    hiveJob: Optional. Job is a Hive job.
    labels: Optional. The labels to associate with this job.Label keys must be
      between 1 and 63 characters long, and must conform to the following
      regular expression: \\p{Ll}\\p{Lo}{0,62}Label values must be between 1 and
      63 characters long, and must conform to the following regular
      expression: \\p{Ll}\\p{Lo}\\p{N}_-{0,63}No more than 32 labels can be
      associated with a given job.
    pigJob: Optional. Job is a Pig job.
    prerequisiteStepIds: Optional. The optional list of prerequisite job
      step_ids. If not specified, the job will start at the beginning of
      workflow.
    prestoJob: Optional. Job is a Presto job.
    pysparkJob: Optional. Job is a PySpark job.
    scheduling: Optional. Job scheduling configuration.
    sparkJob: Optional. Job is a Spark job.
    sparkRJob: Optional. Job is a SparkR job.
    sparkSqlJob: Optional. Job is a SparkSql job.
    stepId: Required. The step id. The id must be unique among all jobs within
      the template.The step id is used as prefix for job id, as job goog-
      dataproc-workflow-step-id label, and in prerequisiteStepIds field from
      other steps.The id must contain only letters (a-z, A-Z), numbers (0-9),
      underscores (_), and hyphens (-). Cannot begin or end with underscore or
      hyphen. Must consist of between 3 and 50 characters.
    trinoJob: Optional. Job is a Trino job.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels to associate with this job.Label keys must be
    between 1 and 63 characters long, and must conform to the following
    regular expression: \\p{Ll}\\p{Lo}{0,62}Label values must be between 1 and
    63 characters long, and must conform to the following regular expression:
    \\p{Ll}\\p{Lo}\\p{N}_-{0,63}No more than 32 labels can be associated with a
    given job.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    flinkJob = _messages.MessageField('FlinkJob', 1)
    hadoopJob = _messages.MessageField('HadoopJob', 2)
    hiveJob = _messages.MessageField('HiveJob', 3)
    labels = _messages.MessageField('LabelsValue', 4)
    pigJob = _messages.MessageField('PigJob', 5)
    prerequisiteStepIds = _messages.StringField(6, repeated=True)
    prestoJob = _messages.MessageField('PrestoJob', 7)
    pysparkJob = _messages.MessageField('PySparkJob', 8)
    scheduling = _messages.MessageField('JobScheduling', 9)
    sparkJob = _messages.MessageField('SparkJob', 10)
    sparkRJob = _messages.MessageField('SparkRJob', 11)
    sparkSqlJob = _messages.MessageField('SparkSqlJob', 12)
    stepId = _messages.StringField(13)
    trinoJob = _messages.MessageField('TrinoJob', 14)