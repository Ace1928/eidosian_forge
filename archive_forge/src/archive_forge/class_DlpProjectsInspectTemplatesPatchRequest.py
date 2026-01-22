from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsInspectTemplatesPatchRequest(_messages.Message):
    """A DlpProjectsInspectTemplatesPatchRequest object.

  Fields:
    googlePrivacyDlpV2UpdateInspectTemplateRequest: A
      GooglePrivacyDlpV2UpdateInspectTemplateRequest resource to be passed as
      the request body.
    name: Required. Resource name of organization and inspectTemplate to be
      updated, for example
      `organizations/433245324/inspectTemplates/432452342` or
      projects/project-id/inspectTemplates/432452342.
  """
    googlePrivacyDlpV2UpdateInspectTemplateRequest = _messages.MessageField('GooglePrivacyDlpV2UpdateInspectTemplateRequest', 1)
    name = _messages.StringField(2, required=True)