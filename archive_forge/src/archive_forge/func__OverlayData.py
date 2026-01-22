from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
def _OverlayData(self, create_data, orig_data):
    """Uses data from the original configuration unless explicitly defined."""
    for k, v in orig_data.items():
        create_data[k] = create_data.get(k) or v
    return create_data