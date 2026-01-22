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
def add_log_denies_enabled(self):
    """Adds handling for log denies enablement."""
    group = self.parser.add_group('Log Denies flags.', mutex=True)
    group.add_argument('--no-log-denies', action='store_true', help='If set, disable all log denies.')
    group.add_argument('--log-denies', action='store_true', help='If set, log all denies and dry run failures. (To disable, use --no-log-denies)')