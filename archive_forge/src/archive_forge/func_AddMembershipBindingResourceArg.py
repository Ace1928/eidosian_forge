from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import util as cmd_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def AddMembershipBindingResourceArg(parser, api_version='v1', binding_help=''):
    """Add resource arg for projects/{}/locations/{}/memberships/{}/bindings/{}."""
    flag_name = 'BINDING'
    spec = concepts.ResourceSpec('gkehub.projects.locations.memberships.bindings', api_version=api_version, resource_name='binding', plural_name='bindings', disable_auto_completers=True, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=_LocationAttributeConfig(), membershipsId=_BasicAttributeConfig('membership', ''), bindingsId=_BasicAttributeConfig('binding', binding_help))
    concept_parsers.ConceptParser.ForResource(flag_name, spec, 'The group of arguments defining a Membership Binding.', plural=False, required=True).AddToParser(parser)