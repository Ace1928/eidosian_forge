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
def GenerateResourceArg(self, method, anchor_arg_name=None, shared_resource_flags=None, group_help=None):
    """Generates only the resource arg (no update flags)."""
    return self._GenerateConceptParser(self._resource_spec, self.attribute_names, repeated=self.repeated, shared_resource_flags=shared_resource_flags, anchor_arg_name=anchor_arg_name, group_help=group_help, is_required=self.IsRequired(method))