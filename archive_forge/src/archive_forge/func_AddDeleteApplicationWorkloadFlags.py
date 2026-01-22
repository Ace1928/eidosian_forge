from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.apphub import utils as apphub_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddDeleteApplicationWorkloadFlags(parser):
    GetApplicationWorkloadResourceArg().AddToParser(parser)
    parser.add_argument('--async', dest='async_', action='store_true', default=False, help='Return immediately, without waiting for the operation in progress to complete.')