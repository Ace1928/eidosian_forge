from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.core.util import files
import six
def _ParseParameterTypeAndValue(param_string):
    """Parse a string of the form <recursive_type>:<value> into each part."""
    type_string, value_string = _SplitParam(param_string)
    if not type_string:
        type_string = 'STRING'
    type_dict = _ParseParameterType(type_string)
    return (type_dict, _ParseParameterValue(type_dict, value_string))