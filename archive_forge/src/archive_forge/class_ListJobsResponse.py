from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListJobsResponse(_messages.Message):
    """ListJobsResponse is a list of Jobs resources.

  Fields:
    apiVersion: The API version for this call such as "run.googleapis.com/v1".
    items: List of Jobs.
    kind: The kind of this resource, in this case "JobsList".
    metadata: Metadata associated with this jobs list.
    unreachable: Locations that could not be reached.
  """
    apiVersion = _messages.StringField(1)
    items = _messages.MessageField('Job', 2, repeated=True)
    kind = _messages.StringField(3)
    metadata = _messages.MessageField('ListMeta', 4)
    unreachable = _messages.StringField(5, repeated=True)