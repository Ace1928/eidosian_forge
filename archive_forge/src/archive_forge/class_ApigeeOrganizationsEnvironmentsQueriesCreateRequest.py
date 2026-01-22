from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsQueriesCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsQueriesCreateRequest object.

  Fields:
    googleCloudApigeeV1Query: A GoogleCloudApigeeV1Query resource to be passed
      as the request body.
    parent: Required. The parent resource name. Must be of the form
      `organizations/{org}/environments/{env}`.
  """
    googleCloudApigeeV1Query = _messages.MessageField('GoogleCloudApigeeV1Query', 1)
    parent = _messages.StringField(2, required=True)