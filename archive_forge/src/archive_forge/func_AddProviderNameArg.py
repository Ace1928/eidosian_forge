from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import googlecloudsdk
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddProviderNameArg(parser):
    """Adds an argument for an Eventarc provider name."""
    parser.add_argument('--name', required=False, help='A provider name (e.g. `storage.googleapis.com`) List results will be filtered on this provider. Only exact match of the provider name is supported.')