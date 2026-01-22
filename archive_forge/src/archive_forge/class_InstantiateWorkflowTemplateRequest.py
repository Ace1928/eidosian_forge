from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstantiateWorkflowTemplateRequest(_messages.Message):
    """A request to instantiate a workflow template.

  Messages:
    ParametersValue: Optional. Map from parameter names to values that should
      be used for those parameters. Values may not exceed 1000 characters.

  Fields:
    parameters: Optional. Map from parameter names to values that should be
      used for those parameters. Values may not exceed 1000 characters.
    requestId: Optional. A tag that prevents multiple concurrent workflow
      instances with the same tag from running. This mitigates risk of
      concurrent instances started due to retries.It is recommended to always
      set this value to a UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier).The tag
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
    version: Optional. The version of workflow template to instantiate. If
      specified, the workflow will be instantiated only if the current version
      of the workflow template has the supplied version.This option cannot be
      used to instantiate a previous version of workflow template.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """Optional. Map from parameter names to values that should be used for
    those parameters. Values may not exceed 1000 characters.

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
    parameters = _messages.MessageField('ParametersValue', 1)
    requestId = _messages.StringField(2)
    version = _messages.IntegerField(3, variant=_messages.Variant.INT32)