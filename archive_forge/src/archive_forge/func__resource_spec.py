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
@property
def _resource_spec(self):
    """Resource spec generated from the YAML."""
    resource_specs = []
    for sub_resource in self._resources:
        if not sub_resource._disable_auto_completers:
            raise ValueError('disable_auto_completers must be True for multitype resource argument [{}]'.format(self.name))
        resource_specs.append(sub_resource._resource_spec)
    return multitype.MultitypeResourceSpec(self.name, *resource_specs)