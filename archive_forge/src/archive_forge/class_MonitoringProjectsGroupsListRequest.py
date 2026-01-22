from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsGroupsListRequest(_messages.Message):
    """A MonitoringProjectsGroupsListRequest object.

  Fields:
    ancestorsOfGroup: A group name. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID] Returns groups that
      are ancestors of the specified group. The groups are returned in order,
      starting with the immediate parent and ending with the most distant
      ancestor. If the specified group has no immediate parent, the results
      are empty.
    childrenOfGroup: A group name. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID] Returns groups whose
      parent_name field contains the group name. If no groups have this
      parent, the results are empty.
    descendantsOfGroup: A group name. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID] Returns the
      descendants of the specified group. This is a superset of the results
      returned by the children_of_group filter, and includes children-of-
      children, and so forth.
    name: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) whose groups
      are to be listed. The format is: projects/[PROJECT_ID_OR_NUMBER]
    pageSize: A positive number that is the maximum number of results to
      return.
    pageToken: If this field is not empty then it must contain the
      next_page_token value returned by a previous call to this method. Using
      this field causes the method to return additional results from the
      previous method call.
  """
    ancestorsOfGroup = _messages.StringField(1)
    childrenOfGroup = _messages.StringField(2)
    descendantsOfGroup = _messages.StringField(3)
    name = _messages.StringField(4, required=True)
    pageSize = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(6)