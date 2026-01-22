from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApisRevisionsDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsApisRevisionsDeleteRequest object.

  Fields:
    name: Required. API proxy revision in the following format:
      `organizations/{org}/apis/{api}/revisions/{rev}`
  """
    name = _messages.StringField(1, required=True)