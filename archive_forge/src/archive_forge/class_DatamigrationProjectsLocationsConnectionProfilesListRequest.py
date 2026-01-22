from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConnectionProfilesListRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConnectionProfilesListRequest object.

  Fields:
    filter: A filter expression that filters connection profiles listed in the
      response. The expression must specify the field name, a comparison
      operator, and the value that you want to use for filtering. The value
      must be a string, a number, or a boolean. The comparison operator must
      be either =, !=, >, or <. For example, list connection profiles created
      this year by specifying **createTime %gt;
      2020-01-01T00:00:00.000000000Z**. You can also filter nested fields. For
      example, you could specify **mySql.username = %lt;my_username%gt;** to
      list all connection profiles configured to connect with a specific
      username.
    orderBy: the order by fields for the result.
    pageSize: The maximum number of connection profiles to return. The service
      may return fewer than this value. If unspecified, at most 50 connection
      profiles will be returned. The maximum value is 1000; values above 1000
      will be coerced to 1000.
    pageToken: A page token, received from a previous `ListConnectionProfiles`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListConnectionProfiles` must match the
      call that provided the page token.
    parent: Required. The parent, which owns this collection of connection
      profiles.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)