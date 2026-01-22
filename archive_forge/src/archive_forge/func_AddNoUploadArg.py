from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.binauthz import arg_parsers
from googlecloudsdk.command_lib.kms import flags as kms_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs as presentation_specs_lib
def AddNoUploadArg(parser):
    """Adds a --no-upload flag to parser."""
    parser.add_argument('--no-upload', action='store_true', default=False, help='Do not upload the generated attestations to the image registry (using Sigstore tag conventions). Note, attestations are never uploaded to the transparency log.')