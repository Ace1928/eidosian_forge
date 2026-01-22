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
def add_key_create_args(parser):
    """Add args for api-keys create command."""
    restriction_group = parser.add_argument_group()
    client_restriction_group = restriction_group.add_mutually_exclusive_group()
    _allowed_referrers_arg(client_restriction_group)
    _allowed_ips_arg(client_restriction_group)
    _allowed_bundle_ids(client_restriction_group)
    _allowed_application(client_restriction_group)
    _api_targets_arg(restriction_group)
    _annotations(parser)