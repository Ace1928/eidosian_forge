from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Developer(_messages.Message):
    """A GoogleCloudApigeeV1Developer object.

  Fields:
    accessType: Access type.
    appFamily: Developer app family.
    apps: List of apps associated with the developer.
    attributes: Optional. Developer attributes (name/value pairs). The custom
      attribute limit is 18.
    companies: List of companies associated with the developer.
    createdAt: Output only. Time at which the developer was created in
      milliseconds since epoch.
    developerId: ID of the developer. **Note**: IDs are generated internally
      by Apigee and are not guaranteed to stay the same over time.
    email: Required. Email address of the developer. This value is used to
      uniquely identify the developer in Apigee hybrid. Note that the email
      address has to be in lowercase only.
    firstName: Required. First name of the developer.
    lastModifiedAt: Output only. Time at which the developer was last modified
      in milliseconds since epoch.
    lastName: Required. Last name of the developer.
    organizationName: Output only. Name of the Apigee organization in which
      the developer resides.
    status: Output only. Status of the developer. Valid values are `active`
      and `inactive`.
    userName: Required. User name of the developer. Not used by Apigee hybrid.
  """
    accessType = _messages.StringField(1)
    appFamily = _messages.StringField(2)
    apps = _messages.StringField(3, repeated=True)
    attributes = _messages.MessageField('GoogleCloudApigeeV1Attribute', 4, repeated=True)
    companies = _messages.StringField(5, repeated=True)
    createdAt = _messages.IntegerField(6)
    developerId = _messages.StringField(7)
    email = _messages.StringField(8)
    firstName = _messages.StringField(9)
    lastModifiedAt = _messages.IntegerField(10)
    lastName = _messages.StringField(11)
    organizationName = _messages.StringField(12)
    status = _messages.StringField(13)
    userName = _messages.StringField(14)