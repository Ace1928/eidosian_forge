from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddRootVolumeSize(parser):
    parser.add_argument('--root-volume-size', type=arg_parsers.BinarySize(suggested_binary_size_scales=['GB', 'GiB', 'TB', 'TiB'], default_unit='Gi'), help='Size of the root volume. The value must be a whole number followed by a size unit of `GB` for gigabyte, or `TB` for terabyte. If no size unit is specified, GB is assumed.')