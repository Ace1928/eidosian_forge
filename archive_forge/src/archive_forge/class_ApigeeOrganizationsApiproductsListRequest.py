from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApiproductsListRequest(_messages.Message):
    """A ApigeeOrganizationsApiproductsListRequest object.

  Fields:
    attributename: Name of the attribute used to filter the search.
    attributevalue: Value of the attribute used to filter the search.
    count: Enter the number of API products you want returned in the API call.
      The limit is 1000.
    expand: Flag that specifies whether to expand the results. Set to `true`
      to get expanded details about each API.
    filter: The filter expression to be used to get the list of API products,
      where filtering can be done on name. Example: filter = "name = foobar"
    pageSize: Count of API products a single page can have in the response. If
      unspecified, at most 100 API products will be returned. The maximum
      value is 100; values above 100 will be coerced to 100.
    pageToken: The starting index record for listing the developers.
    parent: Required. Name of the organization. Use the following structure in
      your request: `organizations/{org}`
    startKey: Gets a list of API products starting with a specific API product
      in the list. For example, if you're returning 50 API products at a time
      (using the `count` query parameter), you can view products 50-99 by
      entering the name of the 50th API product in the first API (without
      using `startKey`). Product name is case sensitive.
  """
    attributename = _messages.StringField(1)
    attributevalue = _messages.StringField(2)
    count = _messages.IntegerField(3)
    expand = _messages.BooleanField(4)
    filter = _messages.StringField(5)
    pageSize = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(7)
    parent = _messages.StringField(8, required=True)
    startKey = _messages.StringField(9)