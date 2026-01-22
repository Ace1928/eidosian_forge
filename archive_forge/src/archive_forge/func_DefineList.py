from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def DefineList(self, field_name, is_operations=False):
    """Defines the List functionality on the calling class.

    Args:
      field_name: The name of the field on the list response to list
      is_operations: Operations have a slightly altered message structure, set
                     to true in operations client
    """

    def List(self, parent_name, filters=None, limit=None, page_size=None, sort_by=None):
        """Lists the objects under a given parent.

      Args:
        self: the self object function will be bound to.
        parent_name: Resource name of the parent to list under.
        filters: Filters to be applied to results (optional).
        limit: Limit to the number of results per page (optional).
        page_size: the number of results per page (optional).
        sort_by: Instructions about how to sort the results (optional).

      Returns:
        List Pager.
      """
        if is_operations:
            req = self.list_request(filter=filters, name=parent_name)
        else:
            req = self.list_request(filter=filters, parent=parent_name, orderBy=sort_by)
        return list_pager.YieldFromList(self.service, req, limit=limit, batch_size_attribute='pageSize', batch_size=page_size, field=field_name)
    setattr(self, 'List', types.MethodType(List, self))