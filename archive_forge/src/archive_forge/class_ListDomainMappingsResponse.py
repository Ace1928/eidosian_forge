from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDomainMappingsResponse(_messages.Message):
    """ListDomainMappingsResponse is a list of DomainMapping resources.

  Fields:
    apiVersion: The API version for this call such as
      "domains.cloudrun.com/v1".
    items: List of DomainMappings.
    kind: The kind of this resource, in this case "DomainMappingList".
    metadata: Metadata associated with this DomainMapping list.
    unreachable: Locations that could not be reached.
  """
    apiVersion = _messages.StringField(1)
    items = _messages.MessageField('DomainMapping', 2, repeated=True)
    kind = _messages.StringField(3)
    metadata = _messages.MessageField('ListMeta', 4)
    unreachable = _messages.StringField(5, repeated=True)