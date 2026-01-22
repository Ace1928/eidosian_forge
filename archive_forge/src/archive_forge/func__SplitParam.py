from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.core.util import files
import six
def _SplitParam(param_string):
    split = param_string.split(':', 1)
    if len(split) != 2:
        raise exceptions.Error('Query parameters must be of the form: "name:type:value", ":type:value", or "name::value". An empty name produces a positional parameter. An empty type produces a STRING parameter.')
    return split