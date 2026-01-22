from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def DefineGet(self):
    """Defines basic get function on an assigned class."""

    def Get(self, object_ref):
        """Gets an object.

      Args:
        self: The self of the class this is set on.
        object_ref: Resource, resource reference for object to get.

      Returns:
        The object requested.
      """
        req = self.get_request(name=object_ref.RelativeName())
        return self.service.Get(req)
    setattr(self, 'Get', types.MethodType(Get, self))