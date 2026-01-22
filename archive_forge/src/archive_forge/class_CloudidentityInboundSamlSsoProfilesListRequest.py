from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityInboundSamlSsoProfilesListRequest(_messages.Message):
    """A CloudidentityInboundSamlSsoProfilesListRequest object.

  Fields:
    filter: A [Common Expression Language](https://github.com/google/cel-spec)
      expression to filter the results. The only supported filter is filtering
      by customer. For example: `customer=="customers/C0123abc"`. Omitting the
      filter or specifying a filter of `customer=="customers/my_customer"`
      will return the profiles for the customer that the caller (authenticated
      user) belongs to.
    pageSize: The maximum number of InboundSamlSsoProfiles to return. The
      service may return fewer than this value. If omitted (or defaulted to
      zero) the server will use a sensible default. This default may change
      over time. The maximum allowed value is 100. Requests with page_size
      greater than that will be silently interpreted as having this maximum
      value.
    pageToken: A page token, received from a previous
      `ListInboundSamlSsoProfiles` call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      `ListInboundSamlSsoProfiles` must match the call that provided the page
      token.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)