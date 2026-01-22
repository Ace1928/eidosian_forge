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
def AddRequiredImportMethodFlag(parser):
    parser.add_argument('--import-method', choices=sorted(maps.IMPORT_METHOD_MAPPER.choices)[1:], help='The wrapping method to be used for incoming key material. For more information about choosing an import method, see https://cloud.google.com/kms/docs/key-wrapping.', required=True)