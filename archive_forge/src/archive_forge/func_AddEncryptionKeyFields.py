from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddEncryptionKeyFields(parser):
    """Adds the --kms-key and --kms-key-service-account flags to the given parser."""
    group = parser.add_group(help='Encryption key settings')
    help_text = '  The customer-managed encryption key to use for this config. If not specified,\n  a Google-managed encryption key is used.\n  '
    group.add_argument('--kms-key', type=str, help=help_text, required=True)
    help_text = '  The service account associated with the provided customer-managed encryption\n  key.\n  '
    group.add_argument('--kms-key-service-account', type=str, help=help_text)