from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as core_resources
def _SetIfNotNone(field_name, field_value, obj, update_mask):
    """Sets specified field to the provided value and adds it to update mask.

  Args:
    field_name: The name of the field to set the value of.
    field_value: The value to set the field to. If it is None, the field will
      NOT be set.
    obj: The object on which the value is to be set.
    update_mask: The update mask to add this field to.

  Returns:
    True if the field was set and False otherwise.
  """
    if field_value is not None:
        setattr(obj, field_name, field_value)
        update_mask.append(field_name)
        return True
    return False