from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OrgpolicyOrganizationsCustomConstraintsGetRequest(_messages.Message):
    """A OrgpolicyOrganizationsCustomConstraintsGetRequest object.

  Fields:
    name: Required. Resource name of the custom constraint. See the custom
      constraint entry for naming requirements.
  """
    name = _messages.StringField(1, required=True)