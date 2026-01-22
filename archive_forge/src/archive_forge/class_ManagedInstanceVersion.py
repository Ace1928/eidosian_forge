from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedInstanceVersion(_messages.Message):
    """A ManagedInstanceVersion object.

  Fields:
    instanceTemplate: [Output Only] The intended template of the instance.
      This field is empty when current_action is one of { DELETING, ABANDONING
      }.
    name: [Output Only] Name of the version.
  """
    instanceTemplate = _messages.StringField(1)
    name = _messages.StringField(2)