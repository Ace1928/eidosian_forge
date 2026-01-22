from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryResourcesCalendarsListRequest(_messages.Message):
    """A DirectoryResourcesCalendarsListRequest object.

  Fields:
    customer: The unique ID for the customer's G Suite account. As an account
      administrator, you can also use the my_customer alias to represent your
      account's customer ID.
    maxResults: Maximum number of results to return.
    orderBy: Field(s) to sort results by in either ascending or descending
      order. Supported fields include resourceId, resourceName, capacity,
      buildingId, and floorName. If no order is specified, defaults to
      ascending. Should be of the form "field [asc|desc], field [asc|desc],
      ...". For example buildingId, capacity desc would return results sorted
      first by buildingId in ascending order then by capacity in descending
      order.
    pageToken: Token to specify the next page in the list.
    query: String query used to filter results. Should be of the form "field
      operator value" where field can be any of supported fields and operators
      can be any of supported operations. Operators include '=' for exact
      match and ':' for prefix match or HAS match where applicable. For prefix
      match, the value should always be followed by a *. Supported fields
      include generatedResourceName, name, buildingId,
      featureInstances.feature.name. For example buildingId=US-NYC-9TH AND
      featureInstances.feature.name:Phone.
  """
    customer = _messages.StringField(1, required=True)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    orderBy = _messages.StringField(3)
    pageToken = _messages.StringField(4)
    query = _messages.StringField(5)