from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class NetworkConfigurationCatalog(_messages.Message):
    """A NetworkConfigurationCatalog object.

  Fields:
    configurations: A NetworkConfiguration attribute.
  """
    configurations = _messages.MessageField('NetworkConfiguration', 1, repeated=True)