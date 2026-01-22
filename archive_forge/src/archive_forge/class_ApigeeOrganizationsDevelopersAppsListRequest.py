from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersAppsListRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersAppsListRequest object.

  Fields:
    count: Number of developer apps to return in the API call. Use with the
      `startKey` parameter to provide more targeted filtering. The limit is
      1000.
    expand: Optional. Specifies whether to expand the results. Set to `true`
      to expand the results. This query parameter is not valid if you use the
      `count` or `startKey` query parameters.
    parent: Required. Name of the developer. Use the following structure in
      your request: `organizations/{org}/developers/{developer_email}`
    shallowExpand: Optional. Specifies whether to expand the results in
      shallow mode. Set to `true` to expand the results in shallow mode.
    startKey: **Note**: Must be used in conjunction with the `count`
      parameter. Name of the developer app from which to start displaying the
      list of developer apps. For example, if you're returning 50 developer
      apps at a time (using the `count` query parameter), you can view
      developer apps 50-99 by entering the name of the 50th developer app. The
      developer app name is case sensitive.
  """
    count = _messages.IntegerField(1)
    expand = _messages.BooleanField(2)
    parent = _messages.StringField(3, required=True)
    shallowExpand = _messages.BooleanField(4)
    startKey = _messages.StringField(5)