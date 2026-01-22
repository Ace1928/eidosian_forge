from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import itertools
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.calliope.concepts import util as resource_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import update_args
from googlecloudsdk.command_lib.util.apis import update_resource_args
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
def _GetFieldFromMethods(self, methods):
    """Gets apitools field associated with api_field from methods."""
    if not methods or not self.api_field:
        return None
    field = self._GetField(methods[0].GetRequestType())
    for method in methods:
        other_field = self._GetField(method.GetRequestType())
        if field.name != other_field.name or field.variant != other_field.variant or field.repeated != other_field.repeated:
            message_names = ', '.join((method.GetRequestType().__name__ for method in methods))
            raise util.InvalidSchemaError(f'Unable to generate flag for api field {self.api_field}. Found non equivalent fields in messages: [{message_names}].')
    return field