from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
import six
def ParseOperationJsonMetadata(metadata_value, metadata_type):
    if not metadata_value:
        return metadata_type()
    return encoding.JsonToMessage(metadata_type, encoding.MessageToJson(metadata_value))