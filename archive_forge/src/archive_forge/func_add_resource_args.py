from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def add_resource_args(parser):
    """Adds resource args for command."""
    resource_group = parser.add_mutually_exclusive_group(required=False)
    _project_id_flag(resource_group)
    _folder_id_flag(resource_group)
    _orgnaization_id_flag(resource_group)