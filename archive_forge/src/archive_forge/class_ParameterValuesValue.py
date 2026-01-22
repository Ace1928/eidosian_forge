from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ParameterValuesValue(_messages.Message):
    """The runtime parameters of the PipelineJob. The parameters will be
    passed into PipelineJob.pipeline_spec to replace the placeholders at
    runtime. This field is used by pipelines built using
    `PipelineJob.pipeline_spec.schema_version` 2.1.0, such as pipelines built
    using Kubeflow Pipelines SDK 1.9 or higher and the v2 DSL.

    Messages:
      AdditionalProperty: An additional property for a ParameterValuesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ParameterValuesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ParameterValuesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)