from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.export import util
from googlecloudsdk.core.console import console_io
def add_referential_rules(self):
    """Adds handling for referential rules enablement."""
    group = self.parser.add_group('Referential Rules flags.', mutex=True)
    group.add_argument('--no-referential-rules', action='store_true', help='Disables referential rules support.')
    group.add_argument('--referential-rules', action='store_true', help='If set, enable support for referential constraints. (To disable, use --no-referential-rules)')