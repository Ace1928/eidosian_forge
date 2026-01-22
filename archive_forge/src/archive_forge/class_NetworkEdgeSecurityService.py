from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkEdgeSecurityService(_messages.Message):
    """Represents a Google Cloud Armor network edge security service resource.

  Fields:
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    fingerprint: Fingerprint of this resource. A hash of the contents stored
      in this object. This field is used in optimistic locking. This field
      will be ignored when inserting a NetworkEdgeSecurityService. An up-to-
      date fingerprint must be provided in order to update the
      NetworkEdgeSecurityService, otherwise the request will fail with error
      412 conditionNotMet. To see the latest fingerprint, make a get() request
      to retrieve a NetworkEdgeSecurityService.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output only] Type of the resource. Always
      compute#networkEdgeSecurityService for NetworkEdgeSecurityServices
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    region: [Output Only] URL of the region where the resource resides. You
      must specify this field as part of the HTTP request URL. It is not
      settable as a field in the request body.
    securityPolicy: The resource URL for the network edge security service
      associated with this network edge security service.
    selfLink: [Output Only] Server-defined URL for the resource.
    selfLinkWithId: [Output Only] Server-defined URL for this resource with
      the resource id.
  """
    creationTimestamp = _messages.StringField(1)
    description = _messages.StringField(2)
    fingerprint = _messages.BytesField(3)
    id = _messages.IntegerField(4, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(5, default='compute#networkEdgeSecurityService')
    name = _messages.StringField(6)
    region = _messages.StringField(7)
    securityPolicy = _messages.StringField(8)
    selfLink = _messages.StringField(9)
    selfLinkWithId = _messages.StringField(10)