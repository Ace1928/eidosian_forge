from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceEnrollmentStatus(_messages.Message):
    """Represents a resource (project or folder) with its enrollment status.

  Fields:
    children: Output only. The children of the current resource.
    enrollment: Output only. Enrollment which contains enrolled destination
      details for a resource
    name: Identifier. The name of this resource.
  """
    children = _messages.MessageField('ResourceEnrollmentStatus', 1, repeated=True)
    enrollment = _messages.MessageField('Enrollment', 2)
    name = _messages.StringField(3)