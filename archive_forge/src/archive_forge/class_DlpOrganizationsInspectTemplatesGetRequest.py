from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpOrganizationsInspectTemplatesGetRequest(_messages.Message):
    """A DlpOrganizationsInspectTemplatesGetRequest object.

  Fields:
    name: Required. Resource name of the organization and inspectTemplate to
      be read, for example
      `organizations/433245324/inspectTemplates/432452342` or
      projects/project-id/inspectTemplates/432452342.
  """
    name = _messages.StringField(1, required=True)