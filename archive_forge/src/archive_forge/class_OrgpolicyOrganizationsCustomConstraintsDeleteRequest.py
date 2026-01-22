from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OrgpolicyOrganizationsCustomConstraintsDeleteRequest(_messages.Message):
    """A OrgpolicyOrganizationsCustomConstraintsDeleteRequest object.

  Fields:
    name: Required. Name of the custom constraint to delete. See the custom
      constraint entry for naming rules.
  """
    name = _messages.StringField(1, required=True)