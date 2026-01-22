from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppAccessCollections(_messages.Message):
    """JSON template for App Access Collections Resource object in Directory

  API.

  Fields:
    blockedApiAccessBuckets: List of blocked api access buckets.
    enforceSettingsForAndroidDrive: Boolean to indicate whether to enforce app
      access settings on Android Drive or not.
    errorMessage: Error message provided by the Admin that will be shown to
      the user when an app is blocked.
    etag: ETag of the resource.
    kind: Identifies the resource as an app access collection. Value:
      admin#directory#appaccesscollection
    resourceId: Unique ID of app access collection. (Readonly)
    resourceName: Resource name given by the customer while creating/updating.
      Should be unique under given customer.
    trustDomainOwnedApps: Boolean that indicates whether to trust domain owned
      apps.
  """
    blockedApiAccessBuckets = _messages.StringField(1, repeated=True)
    enforceSettingsForAndroidDrive = _messages.BooleanField(2)
    errorMessage = _messages.StringField(3)
    etag = _messages.StringField(4)
    kind = _messages.StringField(5, default=u'admin#directory#appaccesscollection')
    resourceId = _messages.IntegerField(6)
    resourceName = _messages.StringField(7)
    trustDomainOwnedApps = _messages.BooleanField(8)