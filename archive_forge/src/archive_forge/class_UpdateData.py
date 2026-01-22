from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
class UpdateData(object):
    """A holder object for yaml update command."""

    def __init__(self, data):
        self.mask_field = data.get('mask_field', None)
        self.read_modify_update = data.get('read_modify_update', False)
        self.disable_auto_field_mask = data.get('disable_auto_field_mask', False)