from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAnalyticsDatastoresTestRequest(_messages.Message):
    """A ApigeeOrganizationsAnalyticsDatastoresTestRequest object.

  Fields:
    googleCloudApigeeV1Datastore: A GoogleCloudApigeeV1Datastore resource to
      be passed as the request body.
    parent: Required. The parent organization name Must be of the form
      `organizations/{org}`
  """
    googleCloudApigeeV1Datastore = _messages.MessageField('GoogleCloudApigeeV1Datastore', 1)
    parent = _messages.StringField(2, required=True)