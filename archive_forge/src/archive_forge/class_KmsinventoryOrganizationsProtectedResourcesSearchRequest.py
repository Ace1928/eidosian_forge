from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class KmsinventoryOrganizationsProtectedResourcesSearchRequest(_messages.Message):
    """A KmsinventoryOrganizationsProtectedResourcesSearchRequest object.

  Fields:
    cryptoKey: Required. The resource name of the CryptoKey.
    pageSize: The maximum number of resources to return. The service may
      return fewer than this value. If unspecified, at most 500 resources will
      be returned. The maximum value is 500; values above 500 will be coerced
      to 500.
    pageToken: A page token, received from a previous
      KeyTrackingService.SearchProtectedResources call. Provide this to
      retrieve the subsequent page. When paginating, all other parameters
      provided to KeyTrackingService.SearchProtectedResources must match the
      call that provided the page token.
    resourceTypes: Optional. A list of resource types that this request
      searches for. If empty, it will search all the [trackable resource
      types](https://cloud.google.com/kms/docs/view-key-usage#tracked-
      resource-types). Regular expressions are also supported. For example: *
      `compute.googleapis.com.*` snapshots resources whose type starts with
      `compute.googleapis.com`. * `.*Image` snapshots resources whose type
      ends with `Image`. * `.*Image.*` snapshots resources whose type contains
      `Image`. See [RE2](https://github.com/google/re2/wiki/Syntax) for all
      supported regular expression syntax. If the regular expression does not
      match any supported resource type, an INVALID_ARGUMENT error will be
      returned.
    scope: Required. Resource name of the organization. Example:
      organizations/123
  """
    cryptoKey = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    resourceTypes = _messages.StringField(4, repeated=True)
    scope = _messages.StringField(5, required=True)