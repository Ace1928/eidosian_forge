import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __AddAdditionalPropertyType(self, name, property_schema):
    """Add a new nested AdditionalProperty message."""
    new_type_name = 'AdditionalProperty'
    property_schema = dict(property_schema)
    property_schema.pop('description', None)
    description = 'An additional property for a %s object.' % name
    schema = {'id': new_type_name, 'type': 'object', 'description': description, 'properties': {'key': {'type': 'string', 'description': 'Name of the additional property.'}, 'value': property_schema}}
    self.AddDescriptorFromSchema(new_type_name, schema)
    return new_type_name