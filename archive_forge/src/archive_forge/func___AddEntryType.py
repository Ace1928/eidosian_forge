import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __AddEntryType(self, entry_type_name, entry_schema, parent_name):
    """Add a type for a list entry."""
    entry_schema.pop('description', None)
    description = 'Single entry in a %s.' % parent_name
    schema = {'id': entry_type_name, 'type': 'object', 'description': description, 'properties': {'entry': {'type': 'array', 'items': entry_schema}}}
    self.AddDescriptorFromSchema(entry_type_name, schema)
    return entry_type_name