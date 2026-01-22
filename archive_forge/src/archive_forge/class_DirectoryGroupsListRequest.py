from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryGroupsListRequest(_messages.Message):
    """A DirectoryGroupsListRequest object.

  Enums:
    OrderByValueValuesEnum: Column to use for sorting results
    SortOrderValueValuesEnum: Whether to return results in ascending or
      descending order. Only of use when orderBy is also used

  Fields:
    customer: Immutable ID of the G Suite account. In case of multi-domain, to
      fetch all groups for a customer, fill this field instead of domain.
    domain: Name of the domain. Fill this field to get groups from only this
      domain. To return all groups in a multi-domain fill customer field
      instead.
    maxResults: Maximum number of results to return. Max allowed value is 200.
    orderBy: Column to use for sorting results
    pageToken: Token to specify next page in the list
    query: Query string search. Should be of the form "". Complete
      documentation is at https://developers.google.com/admin-
      sdk/directory/v1/guides/search-groups
    sortOrder: Whether to return results in ascending or descending order.
      Only of use when orderBy is also used
    userKey: Email or immutable ID of the user if only those groups are to be
      listed, the given user is a member of. If it's an ID, it should match
      with the ID of the user object.
  """

    class OrderByValueValuesEnum(_messages.Enum):
        """Column to use for sorting results

    Values:
      email: Email of the group.
    """
        email = 0

    class SortOrderValueValuesEnum(_messages.Enum):
        """Whether to return results in ascending or descending order.

    Only of
    use when orderBy is also used

    Values:
      ASCENDING: Ascending order.
      DESCENDING: Descending order.
    """
        ASCENDING = 0
        DESCENDING = 1
    customer = _messages.StringField(1)
    domain = _messages.StringField(2)
    maxResults = _messages.IntegerField(3, variant=_messages.Variant.INT32, default=200)
    orderBy = _messages.EnumField('OrderByValueValuesEnum', 4)
    pageToken = _messages.StringField(5)
    query = _messages.StringField(6)
    sortOrder = _messages.EnumField('SortOrderValueValuesEnum', 7)
    userKey = _messages.StringField(8)