from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def DefineDelete(self):
    """Defines basic delete function on an assigned class."""

    def Delete(self, object_ref):
        """Deletes a given object given an object name.

      Args:
        self: The self of the class this is set on.
        object_ref: Resource, resource reference for object to delete.

      Returns:
        Long running operation.
      """
        req = self.delete_request(name=object_ref.RelativeName())
        return self.service.Delete(req)
    setattr(self, 'Delete', types.MethodType(Delete, self))