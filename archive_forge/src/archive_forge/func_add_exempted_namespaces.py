from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
def add_exempted_namespaces(self):
    """Adds handling for configuring exempted namespaces on content bundles."""
    group = self.parser.add_argument_group('Exempted Namespaces flags.', mutex=True)
    group.add_argument('--exempted-namespaces', type=str, help='Exempted namespaces are ignored by Policy Controller when applying constraints added by this bundle.')
    group.add_argument('--no-exempted-namespaces', action='store_true', help='Removes all exempted namespaces from the specified bundle.')