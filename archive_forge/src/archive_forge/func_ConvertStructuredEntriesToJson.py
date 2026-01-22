from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import arg_parsers
def ConvertStructuredEntriesToJson(structured_entries_message):
    structured_entries_dict = {'entries': {}}
    for structured_entry in structured_entries_message.entries.additionalProperties:
        structured_entries_dict['entries'][structured_entry.key] = encoding.MessageToDict(structured_entry.value)
    return json.dumps(structured_entries_dict)