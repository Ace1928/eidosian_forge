from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LaunchTemplateParameters(_messages.Message):
    """Parameters to provide to the template being launched. Note that the
  [metadata in the pipeline code]
  (https://cloud.google.com/dataflow/docs/guides/templates/creating-
  templates#metadata) determines which runtime parameters are valid.

  Messages:
    ParametersValue: The runtime parameters to pass to the job.
    TransformNameMappingValue: Only applicable when updating a pipeline. Map
      of transform name prefixes of the job to be replaced to the
      corresponding name prefixes of the new job.

  Fields:
    environment: The runtime environment for the job.
    jobName: Required. The job name to use for the created job. The name must
      match the regular expression `[a-z]([-a-z0-9]{0,1022}[a-z0-9])?`
    parameters: The runtime parameters to pass to the job.
    transformNameMapping: Only applicable when updating a pipeline. Map of
      transform name prefixes of the job to be replaced to the corresponding
      name prefixes of the new job.
    update: If set, replace the existing pipeline with the name specified by
      jobName with this pipeline, preserving state.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """The runtime parameters to pass to the job.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Additional properties of type ParametersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TransformNameMappingValue(_messages.Message):
        """Only applicable when updating a pipeline. Map of transform name
    prefixes of the job to be replaced to the corresponding name prefixes of
    the new job.

    Messages:
      AdditionalProperty: An additional property for a
        TransformNameMappingValue object.

    Fields:
      additionalProperties: Additional properties of type
        TransformNameMappingValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TransformNameMappingValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    environment = _messages.MessageField('RuntimeEnvironment', 1)
    jobName = _messages.StringField(2)
    parameters = _messages.MessageField('ParametersValue', 3)
    transformNameMapping = _messages.MessageField('TransformNameMappingValue', 4)
    update = _messages.BooleanField(5)