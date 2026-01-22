from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AncestorGroup(_messages.Message):
    """A service ancestor group for which the service is a descendant.

  Fields:
    groupName: Output only. The name of the ancestor group.
    parent: Output only. The parent service which this ancestor contains.
  """
    groupName = _messages.StringField(1)
    parent = _messages.StringField(2)