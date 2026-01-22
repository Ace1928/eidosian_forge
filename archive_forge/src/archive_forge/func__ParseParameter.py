from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.core.util import files
import six
def _ParseParameter(param_string):
    name, param_string = _SplitParam(param_string)
    type_dict, value_dict = _ParseParameterTypeAndValue(param_string)
    result = collections.OrderedDict()
    if name:
        result['name'] = name
    result['parameterType'] = type_dict
    result['parameterValue'] = value_dict
    return result