from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
def _InputMappingFrom(messages, input_mapping_data):
    """Translate a dict of input mapping data into a message object.

  Args:
    messages: The API message to use.
    input_mapping_data: A dict containing input mapping data.

  Returns:
    An InputMapping message object derived from options_data.
  """
    location = input_mapping_data.get('location', None)
    if location is not None:
        location = messages.InputMapping.LocationValueValuesEnum(location)
    return messages.InputMapping(fieldName=input_mapping_data.get('fieldName', None), location=location, methodMatch=input_mapping_data.get('methodMatch', None), value=input_mapping_data.get('value', None))