from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1DumpItem(_messages.Message):
    """Wrapper for any item that can be contained in the dump.

  Fields:
    taggedEntry: Entry and its tags.
  """
    taggedEntry = _messages.MessageField('GoogleCloudDatacatalogV1TaggedEntry', 1)