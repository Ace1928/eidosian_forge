from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddVolumeThroughput(parser, prefix):
    parser.add_argument('--{}-volume-throughput'.format(prefix), type=int, help='Throughput to provision for the {} volume, in MiB/s. Only valid if the volume type is GP3. If volume type is GP3 and throughput is not provided, it defaults to 125.'.format(prefix))