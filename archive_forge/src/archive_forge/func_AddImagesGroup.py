from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddImagesGroup(parser, hidden=False):
    """Adds Images flag."""
    images_group = parser.add_mutually_exclusive_group()
    images_group.add_argument('--images', metavar='NAME=TAG', type=arg_parsers.ArgDict(), hidden=hidden, help=textwrap.dedent('      Reference to a collection of individual image name to image full path replacements.\n\n      For example:\n\n          $ gcloud deploy releases create foo \\\n              --images image1=path/to/image1:v1@sha256:45db24\n      '))
    images_group.add_argument('--build-artifacts', hidden=hidden, help="Reference to a Skaffold build artifacts output file from skaffold build --file-output=BUILD_ARTIFACTS. If you aren't using Skaffold, use the --images flag below to specify the image-names-to-tagged-image references.")