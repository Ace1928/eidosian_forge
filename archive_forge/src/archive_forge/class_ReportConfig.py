from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportConfig(_messages.Message):
    """Message describing ReportConfig object. ReportConfig is the
  configuration to generate reports. See
  https://cloud.google.com/storage/docs/insights/using-inventory-
  reports#create-config-rest for more details on how to set various fields.
  Next ID: 12

  Messages:
    LabelsValue: Labels as key value pairs

  Fields:
    createTime: Output only. [Output only] Create time stamp
    csvOptions: Options for CSV formatted reports.
    displayName: User provided display name which can be empty and limited to
      256 characters that is editable.
    frequencyOptions: The frequency of report generation.
    labels: Labels as key value pairs
    name: name of resource. It will be of form
      projects//locations//reportConfigs/.
    objectMetadataReportOptions: Report for exporting object metadata.
    parquetOptions: Options for Parquet formatted reports.
    updateTime: Output only. [Output only] Update time stamp
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels as key value pairs

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
    createTime = _messages.StringField(1)
    csvOptions = _messages.MessageField('CSVOptions', 2)
    displayName = _messages.StringField(3)
    frequencyOptions = _messages.MessageField('FrequencyOptions', 4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    objectMetadataReportOptions = _messages.MessageField('ObjectMetadataReportOptions', 7)
    parquetOptions = _messages.MessageField('ParquetOptions', 8)
    updateTime = _messages.StringField(9)