from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectsValueListEntry(_messages.Message):
    """A ProjectsValueListEntry object.

    Fields:
      friendlyName: A descriptive name for this project.
      id: An opaque ID of this project.
      kind: The resource type.
      numericId: The numeric ID of this project.
      projectReference: A unique reference to this project.
    """
    friendlyName = _messages.StringField(1)
    id = _messages.StringField(2)
    kind = _messages.StringField(3, default=u'bigquery#project')
    numericId = _messages.IntegerField(4, variant=_messages.Variant.UINT64)
    projectReference = _messages.MessageField('ProjectReference', 5)