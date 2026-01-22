from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApisRevisionsUpdateApiProxyRevisionRequest(_messages.Message):
    """A ApigeeOrganizationsApisRevisionsUpdateApiProxyRevisionRequest object.

  Fields:
    googleApiHttpBody: A GoogleApiHttpBody resource to be passed as the
      request body.
    name: Required. API proxy revision to update in the following format:
      `organizations/{org}/apis/{api}/revisions/{rev}`
    validate: Ignored. All uploads are validated regardless of the value of
      this field. Maintained for compatibility with Apigee Edge API.
  """
    googleApiHttpBody = _messages.MessageField('GoogleApiHttpBody', 1)
    name = _messages.StringField(2, required=True)
    validate = _messages.BooleanField(3)