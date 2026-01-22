from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PolicyNetwork(_messages.Message):
    """A PolicyNetwork object.

  Fields:
    kind: A string attribute.
    networkUrl: The fully qualified URL of the VPC network to bind to. This
      should be formatted like https://www.googleapis.com/compute/v1/projects/
      {project}/global/networks/{network}
  """
    kind = _messages.StringField(1, default='dns#policyNetwork')
    networkUrl = _messages.StringField(2)