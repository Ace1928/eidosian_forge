from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocateIdsResponse(_messages.Message):
    """The response for Datastore.AllocateIds.

  Fields:
    keys: The keys specified in the request (in the same order), each with its
      key path completed with a newly allocated ID.
  """
    keys = _messages.MessageField('Key', 1, repeated=True)