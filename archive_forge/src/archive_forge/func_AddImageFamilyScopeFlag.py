from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
def AddImageFamilyScopeFlag(parser):
    """Add the image-family-scope flag."""
    parser.add_argument('--image-family-scope', metavar='IMAGE_FAMILY_SCOPE', choices=['zonal', 'global'], help='      Sets the scope for the `--image-family` flag. By default, when\n      specifying an image family in a public image project, the zonal image\n      family scope is used. All other projects default to the global\n      image. Use this flag to override this behavior.')