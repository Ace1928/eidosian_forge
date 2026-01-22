from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class EnvVariablesValue(_messages.Message):
    """Optional. Environment variables to provide to the processes executing
    the workflow. Environment variable names must match the regular expression
    "a-zA-Z_*". They cannot specify Apache Airflow configuration overrides
    (they cannot match the regular expression
    `AIRFLOW__[A-Z0-9_]+__[A-Z0-9_]+`)

    Messages:
      AdditionalProperty: An additional property for a EnvVariablesValue
        object.

    Fields:
      additionalProperties: Additional properties of type EnvVariablesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a EnvVariablesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)