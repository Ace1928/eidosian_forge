from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def SetDictDottedKeyUpperCase(input_dict, dotted_key):
    *key, last = dotted_key.split('.')
    for bit in key:
        if bit not in input_dict:
            return
        input_dict = input_dict.get(bit)
    if last in input_dict:
        input_dict[last] = input_dict[last].upper()