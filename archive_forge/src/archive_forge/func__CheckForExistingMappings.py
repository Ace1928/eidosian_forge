import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _CheckForExistingMappings(mapping_type, message_type, python_name, json_name):
    """Validate that no mappings exist for the given values."""
    if mapping_type == 'field':
        getter = GetCustomJsonFieldMapping
    elif mapping_type == 'enum':
        getter = GetCustomJsonEnumMapping
    remapping = getter(message_type, python_name=python_name)
    if remapping is not None and remapping != json_name:
        raise exceptions.InvalidDataError('Cannot add mapping for %s "%s", already mapped to "%s"' % (mapping_type, python_name, remapping))
    remapping = getter(message_type, json_name=json_name)
    if remapping is not None and remapping != python_name:
        raise exceptions.InvalidDataError('Cannot add mapping for %s "%s", already mapped to "%s"' % (mapping_type, json_name, remapping))