from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrefixNode(_messages.Message):
    """A message representing a key prefix node in the key prefix hierarchy.
  for eg. Bigtable keyspaces are lexicographically ordered mappings of keys to
  values. Keys often have a shared prefix structure where users use the keys
  to organize data. Eg ///employee In this case Keysight will possibly use one
  node for a company and reuse it for all employees that fall under the
  company. Doing so improves legibility in the UI.

  Fields:
    dataSourceNode: Whether this corresponds to a data_source name.
    depth: The depth in the prefix hierarchy.
    endIndex: The index of the end key bucket of the range that this node
      spans.
    startIndex: The index of the start key bucket of the range that this node
      spans.
    word: The string represented by the prefix node.
  """
    dataSourceNode = _messages.BooleanField(1)
    depth = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    endIndex = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    startIndex = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    word = _messages.StringField(5)