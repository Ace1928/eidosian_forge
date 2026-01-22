from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Ancestor(_messages.Message):
    """Identifying information for a single ancestor of a project.

  Fields:
    resourceId: Resource id of the ancestor.
  """
    resourceId = _messages.MessageField('ResourceId', 1)