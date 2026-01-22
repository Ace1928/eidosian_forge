from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApisRevisionsGetRequest(_messages.Message):
    """A ApigeeOrganizationsApisRevisionsGetRequest object.

  Fields:
    format: Format used when downloading the API proxy configuration revision.
      Set to `bundle` to download the API proxy configuration revision as a
      zip file.
    name: Required. API proxy revision in the following format:
      `organizations/{org}/apis/{api}/revisions/{rev}`
  """
    format = _messages.StringField(1)
    name = _messages.StringField(2, required=True)