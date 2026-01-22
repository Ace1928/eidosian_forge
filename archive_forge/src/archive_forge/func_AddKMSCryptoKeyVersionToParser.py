from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddKMSCryptoKeyVersionToParser(parser, hidden):
    parser.add_argument('--kms-crypto-key-version', type=str, help='\n      Resource ID of a KMS CryptoKeyVersion used to encrypt the initial password.\n\n      https://cloud.google.com/kms/docs/resource-hierarchy#key_versions\n      ', hidden=hidden)