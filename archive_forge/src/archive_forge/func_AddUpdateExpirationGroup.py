from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.secrets import completers as secrets_completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def AddUpdateExpirationGroup(parser):
    """Add flags for specifying expiration on secret updates.."""
    group = parser.add_group(mutex=True, help='Expiration.')
    group.add_argument(_ArgOrFlag('expire-time', False), metavar='EXPIRE-TIME', help='Timestamp at which to automatically delete the secret.')
    group.add_argument(_ArgOrFlag('ttl', False), metavar='TTL', help='Duration of time (in seconds) from the running of the command until the secret is automatically deleted.')
    group.add_argument(_ArgOrFlag('remove-expiration', False), action='store_true', help='If set, removes scheduled expiration from secret (if it had one).')