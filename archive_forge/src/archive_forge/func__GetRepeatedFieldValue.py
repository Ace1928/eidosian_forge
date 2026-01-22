from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import perimeters
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
def _GetRepeatedFieldValue(args, field_name, base_config_value, has_spec):
    """Returns the repeated field value to use for the update operation."""
    repeated_field = repeated.ParsePrimitiveArgs(args, field_name, lambda: base_config_value or [])
    if not has_spec and (not repeated_field):
        repeated_field = base_config_value
    return repeated_field