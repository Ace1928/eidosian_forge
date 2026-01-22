from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DescendantService(_messages.Message):
    """A service group descendant that belongs to a certain service group
  directly or to one of its included groups.

  Fields:
    parent: Output only. The parent group to which this descendant belongs.
    serviceName: Output only. The name of the descendant service.
  """
    parent = _messages.StringField(1)
    serviceName = _messages.StringField(2)