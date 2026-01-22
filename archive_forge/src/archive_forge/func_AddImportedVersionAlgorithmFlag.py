from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
def AddImportedVersionAlgorithmFlag(parser):
    parser.add_argument('--algorithm', choices=sorted(maps.ALGORITHMS_FOR_IMPORT), help='The algorithm to assign to the new key version. For more information about supported algorithms, see https://cloud.google.com/kms/docs/algorithms.', required=True)