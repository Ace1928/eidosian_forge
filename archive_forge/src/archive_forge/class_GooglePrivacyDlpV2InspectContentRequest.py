from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InspectContentRequest(_messages.Message):
    """Request to search for potentially sensitive info in a ContentItem.

  Fields:
    inspectConfig: Configuration for the inspector. What specified here will
      override the template referenced by the inspect_template_name argument.
    inspectTemplateName: Template to use. Any configuration directly specified
      in inspect_config will override those set in the template. Singular
      fields that are set in this request will replace their corresponding
      fields in the template. Repeated fields are appended. Singular sub-
      messages and groups are recursively merged.
    item: The item to inspect.
    locationId: Deprecated. This field has no effect.
  """
    inspectConfig = _messages.MessageField('GooglePrivacyDlpV2InspectConfig', 1)
    inspectTemplateName = _messages.StringField(2)
    item = _messages.MessageField('GooglePrivacyDlpV2ContentItem', 3)
    locationId = _messages.StringField(4)