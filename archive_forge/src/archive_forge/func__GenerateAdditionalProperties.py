from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def _GenerateAdditionalProperties(values_dict):
    """Format values_dict into additionalProperties-style dict."""
    props = [{'key': k, 'value': v} for k, v in sorted(values_dict.items())]
    return {'additionalProperties': props}