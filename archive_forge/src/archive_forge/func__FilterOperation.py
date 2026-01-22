from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.alloydb import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _FilterOperation(operation_item):
    if args.cluster is None:
        return True
    for additional_property in operation_item.metadata.additionalProperties:
        if additional_property.key == 'target':
            target = additional_property.value.string_value
            return self._matchesTarget(target, args.cluster)
    return False